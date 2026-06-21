// File: ResamplingContainer.h
// Created Date: Saturday December 16th 2023
// Author: Steven Atkinson (steven@atkinson.mn)

// A container for real-time resampling using a Lanczos anti-aliasing filter

// This file originally came from the iPlug2 library and has been subsequently modified;
// the following license is copied as required from
// https://github.com/iPlug2/iPlug2/blob/40ebb560eba68f096221e99ef0ae826611fc2bda/LICENSE.txt
// -------------------------------------------------------------------------------------

/*
iPlug 2 C++ Plug-in Framework.

Copyright (C) the iPlug 2 Developers. Portions copyright other contributors, see each source file for more information.

Based on WDL-OL/iPlug by Oli Larkin (2011-2018), and the original iPlug v1 (2008) by John Schwartz / Cockos

LICENSE:

This software is provided 'as-is', without any express or implied warranty.  In no event will the authors be held liable
for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it
and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If
you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not
required.
1. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original
software.
1. This notice may not be removed or altered from any source distribution.

iPlug 2 includes the following 3rd party libraries (see each license info):

* Cockos WDL https://www.cockos.com/wdl
* NanoVG https://github.com/memononen/nanovg
* NanoSVG https://github.com/memononen/nanosvg
* MetalNanoVG https://github.com/ollix/MetalNanoVG
* RTAudio https://www.music.mcgill.ca/~gary/rtaudio
* RTMidi https://www.music.mcgill.ca/~gary/rtmidi
*/
// -------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <functional>
#include <cmath>
#include <algorithm>
#include <complex>
#include <vector>
#include <chrono>
#include <mutex>
#include <sstream>
#include <fstream>
#include <atomic>
#include <array>
#include <cstdlib>
#include <type_traits>

#if defined(__AVX__)
  #include <immintrin.h>
#elif defined(__SSE2__)
  #include <immintrin.h>
#elif defined(__ARM_NEON__)
  #include <arm_neon.h>
#endif

// #include "IPlugPlatform.h"

// #include "heapbuf.h"
#include "Dependencies/WDL/ptrlist.h"

#include "Dependencies/LanczosResampler.h"

namespace dsp
{

enum class EAntiAliasFilterPhase
{
  MinimumPhaseCascadedFIR,
  LinearCascadedFIRShort,
  LinearCascadedFIRLong
};

// Dot product of two float arrays with SIMD acceleration (AVX / SSE2 / NEON).
// Used for the dense FIR steady-state path in half-band decimation.
// Requires forward-ordered coefficients (reversed vs. the standard lag-k convention)
// and forward-ordered samples starting at (sample_ptr - numTaps + 1).
static inline float DenseFIRDotProduct(const float* coefficients, const float* samples, size_t count)
{
  float result = 0.0f;
  size_t i = 0;
#if defined(__AVX__)
  __m256 sum = _mm256_setzero_ps();
  for (; i + 7 < count; i += 8)
  {
    const __m256 c = _mm256_loadu_ps(coefficients + i);
    const __m256 x = _mm256_loadu_ps(samples + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(c, x));
  }
  // Horizontal reduction
  const __m128 lo = _mm256_castps256_ps128(sum);
  const __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 s4 = _mm_add_ps(lo, hi);
  s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
  result = _mm_cvtss_f32(s4);
#elif defined(__SSE2__)
  __m128 sum0 = _mm_setzero_ps();
  __m128 sum1 = _mm_setzero_ps();
  for (; i + 7 < count; i += 8)
  {
    sum0 = _mm_add_ps(sum0, _mm_mul_ps(_mm_loadu_ps(coefficients + i),     _mm_loadu_ps(samples + i)));
    sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(coefficients + i + 4), _mm_loadu_ps(samples + i + 4)));
  }
  __m128 s4 = _mm_add_ps(sum0, sum1);
  s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
  result = _mm_cvtss_f32(s4);
#elif defined(__ARM_NEON__)
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  for (; i + 7 < count; i += 8)
  {
    sum0 = vmlaq_f32(sum0, vld1q_f32(samples + i),     vld1q_f32(coefficients + i));
    sum1 = vmlaq_f32(sum1, vld1q_f32(samples + i + 4), vld1q_f32(coefficients + i + 4));
  }
  float32x4_t s4 = vaddq_f32(sum0, sum1);
  float32x2_t s2 = vadd_f32(vget_low_f32(s4), vget_high_f32(s4));
  result = vget_lane_f32(vpadd_f32(s2, s2), 0);
#endif
  // Scalar tail (remainder after SIMD lanes)
  for (; i < count; i++)
    result += coefficients[i] * samples[i];
  return result;
}

/** A multi-channel real-time resampling container that can be used to resample
 * audio processing to a specified sample rate for the situation where you have
 * some arbitary DSP code that requires a specific sample rate, then back to
 * the original external sample rate, encapsulating the arbitrary DSP code.

 * Three modes are supported:
 * - Linear interpolation: simple linear interpolation between samples
 * - Cubic interpolation: cubic interpolation between samples
 * - Lanczos: Lanczos resampling uses an approximation of the sinc function to
 *   interpolate between samples. This is the highest quality resampling mode.
 *
 * The Lanczos resampler has a configurable filter size (A) that affects the
 * latency of the resampler. It can also optionally use SIMD instructions to
 * when T==float.
 *
 *
 * @tparam T the sampletype
 * @tparam NCHANS the number of channels
 * @tparam A The Lanczos filter size for the LanczosResampler resampler mode
   A higher value makes the filter closer to an
   ideal stop-band that rejects high-frequency content (anti-aliasing),
   but at the expense of higher latency
 */
template <typename T = double, int NCHANS = 2, size_t A = 12>
class ResamplingContainer
{
public:
  using BlockProcessFunc = std::function<void(T**, T**, int)>;
  using LanczosResamplerType = LanczosResampler<T, NCHANS, A>;

  // :param renderingSampleRate: The sample rate required by the code to be encapsulated.
  // :param bandwidthSampleRate: The maximum useful sample rate of the encapsulated DSP. This limits the reconstruction
  // bandwidth when the external context runs at a higher sample rate than the model.
  ResamplingContainer(double renderingSampleRate,
                      EAntiAliasFilterPhase filterPhase = EAntiAliasFilterPhase::MinimumPhaseCascadedFIR,
                      double bandwidthSampleRate = 0.0)
  : mRenderingSampleRate(renderingSampleRate)
  , mBandwidthSampleRate(bandwidthSampleRate > 0.0 ? bandwidthSampleRate : renderingSampleRate)
  , mFilterPhase(filterPhase)
  {
  }

  ResamplingContainer(const ResamplingContainer&) = delete;
  ResamplingContainer& operator=(const ResamplingContainer&) = delete;

  void SetAntiAliasFilterPhase(EAntiAliasFilterPhase filterPhase)
  {
    mFilterPhase = filterPhase;
  }

  // :param inputSampleRate: The external sample rate interacting with this object.
  // :param blockSize: The largest block size that will be given to this class to process until Reset()  is called
  //     again.
  
void Reset(double inputSampleRate, int blockSize)
  {
    if (mInputSampleRate == inputSampleRate && mMaxBlockSize == blockSize && mDesignedFilterPhase == mFilterPhase)
    {
      ClearBuffers();
      return;
    }

    mInputSampleRate = inputSampleRate;
    mRatio1 = mInputSampleRate / mRenderingSampleRate;
    mRatio2 = mRenderingSampleRate / mInputSampleRate;

    const double exactDownsampleFactor = mRenderingSampleRate / mInputSampleRate;
    mIntegerDownsampleFactor = static_cast<int>(std::round(exactDownsampleFactor));
    mUseIntegerDownsampler =
      mIntegerDownsampleFactor > 1 && std::abs(exactDownsampleFactor - mIntegerDownsampleFactor) < 1.0e-9;

    mUseCascadedHalfBandResampler =
      UsesCascadedFIR() && mUseIntegerDownsampler && IsPowerOfTwo(mIntegerDownsampleFactor);

    // Realtime Minimum Phase path: IIR half-band upsample, NAM at high rate,
    // then true-polyphase IIR half-band decimation by default.
    //
    // Set NAM_MINPHASE_IIR_CLEAN_FUSED=1 to use the previous verified
    // CLEAN_FUSED Butterworth fallback for A/B regression testing.
    mUseRealtimeIIRHalfBand = mUseCascadedHalfBandResampler && UsesMinimumPhaseCascadedFIR();


    mDecimationPhase = 0;

    mMaxBlockSize = blockSize;
    mMaxEncapsulatedBlockSize = MaxEncapsulatedBlockSize(blockSize);

    mScratchExternalInputData.Resize(mMaxBlockSize * NCHANS);
    mEncapsulatedInputData.Resize(mMaxEncapsulatedBlockSize * NCHANS);
    mEncapsulatedOutputData.Resize(mMaxEncapsulatedBlockSize * NCHANS);
    mAntiAliasOutputData.Resize(mMaxEncapsulatedBlockSize * NCHANS);

    mScratchExternalInputPointers.Empty();
    mEncapsulatedInputPointers.Empty();
    mEncapsulatedOutputPointers.Empty();
    mAntiAliasOutputPointers.Empty();

    for (auto chan = 0; chan < NCHANS; chan++)
    {
      mScratchExternalInputPointers.Add(mScratchExternalInputData.Get() + (chan * mMaxBlockSize));
      mEncapsulatedInputPointers.Add(mEncapsulatedInputData.Get() + (chan * mMaxEncapsulatedBlockSize));
      mEncapsulatedOutputPointers.Add(mEncapsulatedOutputData.Get() + (chan * mMaxEncapsulatedBlockSize));
      mAntiAliasOutputPointers.Add(mAntiAliasOutputData.Get() + (chan * mMaxEncapsulatedBlockSize));
    }

    if (mUseCascadedHalfBandResampler)
    {
      // Critical: design after the final selected filter phase is already set.
      // This keeps the actual FIR length and the reported latency in sync.
      DesignCascadedHalfBandAntiAliasFilter();
      mDesignedFilterPhase = mFilterPhase;
      mResampler1 = nullptr;
      mResampler2 = nullptr;
    }
    else
    {
      // Fractional/non-power-of-two fallback. These are internal compatibility paths,
      // not user-selectable legacy filter modes.
      DesignAntiAliasFilter();
      mResampler1 = std::make_unique<LanczosResamplerType>(mInputSampleRate, mRenderingSampleRate);
      mResampler2 =
        mUseIntegerDownsampler ? nullptr : std::make_unique<LanczosResamplerType>(mRenderingSampleRate, mInputSampleRate);
    }

    ClearBuffers();

    if (mUseCascadedHalfBandResampler)
    {
      mLatency = UsesMinimumPhaseCascadedFIR() ? 0 : GetCascadedHalfBandRoundTripLatency();

      if (!UsesMinimumPhaseCascadedFIR() && mLatency <= 0)
        throw std::runtime_error("Linear cascaded FIR selected but reported latency is zero.");

      return;
    }

    const auto midSamples =
      mUseIntegerDownsampler ? static_cast<size_t>(mIntegerDownsampleFactor) : mResampler2->GetNumSamplesRequiredFor(1);

    mLatency = int(mResampler1->GetNumSamplesRequiredFor(midSamples));

    // Fractional fallback latency for the internal compatibility paths.
    if (mAntiAliasEnabled && !UsesMinimumPhaseCascadedFIR())
      mLatency += 64;

    mResampler1->PushBlock(mScratchExternalInputPointers.GetList(), mLatency);
    const size_t populated = mResampler1->PopBlock(mEncapsulatedInputPointers.GetList(), midSamples);

    if (populated < midSamples)
      throw std::runtime_error("Didn't get enough samples required for pre-population!");

    FallbackFunc(mEncapsulatedInputPointers.GetList(), mEncapsulatedOutputPointers.GetList(), (int) populated);

    T** resamplerInput = PrepareDownsamplerInput(populated);

    if (mUseIntegerDownsampler)
      DecimateBlock(resamplerInput, populated, nullptr, 0);
    else
      mResampler2->PushBlock(resamplerInput, populated);
  }

  /** Resample an input block with a per-block function (up sample input -> process with function -> down sample)
   * @param inputs Two-dimensional array containing the non-interleaved input buffers of audio samples for all channels
   * @param outputs Two-dimensional array for audio output (non-interleaved).
   * @param nFrames The block size for this block: number of samples per channel.
   * @param func The function that processes the audio sample at the higher sampling rate. NOTE: std::function can call
   * malloc if you pass in captures */
  
  void ProcessBlock(T** inputs, T** outputs, int nFrames, BlockProcessFunc func)
  {
    if (mUseRealtimeIIRHalfBand)
    {
      ProcessBlockRealtimeIIRHalfBand(inputs, outputs, nFrames, func);
      return;
    }

    if (mUseCascadedHalfBandResampler)
    {
      ProcessBlockLinearCascadedFIR(inputs, outputs, nFrames, func);
      return;
    }

    mResampler1->PushBlock(PrepareUpsamplerInput(inputs, static_cast<size_t>(nFrames)), nFrames);
    mOutputWritePos = 0;

    // This is the most samples the encapsulated context might get. Sometimes it will get fewer.
    const auto maxEncapsulatedLen = MaxEncapsulatedBlockSize(nFrames);

    // Process as much audio as possible with the encapsulated DSP, and push it into the second resampler.
    while (mResampler1->GetNumSamplesRequiredFor(1) == 0)
    {
      const size_t populated1 = mResampler1->PopBlock(mEncapsulatedInputPointers.GetList(), maxEncapsulatedLen);

      if (populated1 > maxEncapsulatedLen)
      {
        throw std::runtime_error("Got more encapsulated samples than the encapsulated DSP is prepared to handle!");
      }

      func(mEncapsulatedInputPointers.GetList(), mEncapsulatedOutputPointers.GetList(), (int) populated1);

      T** resamplerInput = PrepareDownsamplerInput(populated1);

      if (mUseIntegerDownsampler)
        DecimateBlock(resamplerInput, populated1, outputs, nFrames);
      else
        mResampler2->PushBlock(resamplerInput, populated1);
    }

    const auto populated2 = mUseIntegerDownsampler ? mOutputWritePos : mResampler2->PopBlock(outputs, nFrames);

    if (populated2 < static_cast<size_t>(nFrames))
    {
      std::cerr << "Did not yield enough samples (" << populated2 << ") to provide the required output buffer (expected "
                << nFrames << ")! Filling with last sample..." << std::endl;

      for (int c = 0; c < NCHANS; c++)
      {
        const T lastSample = populated2 > 0 ? outputs[c][populated2 - 1] : T(0.0);
        for (int i = static_cast<int>(populated2); i < nFrames; i++)
        {
          outputs[c][i] = lastSample;
        }
      }
    }

    mResampler1->RenormalizePhases();

    if (mResampler2 != nullptr)
      mResampler2->RenormalizePhases();
  }




























  int GetLatency() const { return mLatency; }

private:
  struct OnePoleAllpassState
  {
    // This is actually a 2-sample all-pass section state. The name is kept
    // to avoid touching all existing member declarations.
    T x1 = T(0.0);
    T x2 = T(0.0);
    T y1 = T(0.0);
    T y2 = T(0.0);
  };

  static constexpr int kIIRHalfBandSections = 6;



  static inline int LinearInterpolate(T** inputs, T** outputs, int inputLen, double ratio, int maxOutputLen)
  {
    // FIXME check through this!
    const auto outputLen = std::min(static_cast<int>(std::ceil(static_cast<double>(inputLen) / ratio)), maxOutputLen);

    for (auto writePos = 0; writePos < outputLen; writePos++)
    {
      const auto readPos = ratio * static_cast<double>(writePos);
      const auto readPostionTrunc = std::floor(readPos);
      const auto readPosInt = static_cast<int>(readPostionTrunc);

      if (readPosInt < inputLen)
      {
        const auto y = readPos - readPostionTrunc;

        for (auto chan = 0; chan < NCHANS; chan++)
        {
          const auto x0 = inputs[chan][readPosInt];
          const auto x1 = ((readPosInt + 1) < inputLen) ? inputs[chan][readPosInt + 1] : inputs[chan][readPosInt - 1];
          outputs[chan][writePos] = (1.0 - y) * x0 + y * x1;
        }
      }
    }

    return outputLen;
  }

  static inline int CubicInterpolate(T** inputs, T** outputs, int inputLen, double ratio, int maxOutputLen)
  {
    // FIXME check through this!
    const auto outputLen = std::min(static_cast<int>(std::ceil(static_cast<double>(inputLen) / ratio)), maxOutputLen);

    for (auto writePos = 0; writePos < outputLen; writePos++)
    {
      const auto readPos = ratio * static_cast<double>(writePos);
      const auto readPostionTrunc = std::floor(readPos);
      const auto readPosInt = static_cast<int>(readPostionTrunc);

      if (readPosInt < inputLen)
      {
        const auto y = readPos - readPostionTrunc;

        for (auto chan = 0; chan < NCHANS; chan++)
        {
          const auto xm1 = ((readPosInt - 1) > 0) ? inputs[chan][readPosInt - 1] : 0.0f;
          const auto x0 = ((readPosInt) < inputLen) ? inputs[chan][readPosInt] : inputs[chan][readPosInt - 1];
          const auto x1 = ((readPosInt + 1) < inputLen) ? inputs[chan][readPosInt + 1] : inputs[chan][readPosInt - 1];
          const auto x2 = ((readPosInt + 2) < inputLen) ? inputs[chan][readPosInt + 2] : inputs[chan][readPosInt - 1];

          const auto c = (x1 - xm1) * 0.5;
          const auto v = x0 - x1;
          const auto w = c + v;
          const auto a = w + v + (x2 - x0) * 0.5;
          const auto b = w + a;

          outputs[chan][writePos] = ((((a * y) - b) * y + c) * y + x0);
        }
      }
    }

    return outputLen;
  }

  
void ClearBuffers()
  {
    memset(mScratchExternalInputData.Get(), 0.0f, DataSize(mMaxBlockSize));

    const auto encapsulatedDataSize = DataSize(mMaxEncapsulatedBlockSize);
    memset(mEncapsulatedInputData.Get(), 0.0f, encapsulatedDataSize);
    memset(mEncapsulatedOutputData.Get(), 0.0f, encapsulatedDataSize);
    memset(mAntiAliasOutputData.Get(), 0.0f, encapsulatedDataSize);

    std::fill(mAntiAliasHistory.begin(), mAntiAliasHistory.end(), T(0.0));
    std::fill(mDecimationFirHistory.begin(), mDecimationFirHistory.end(), T(0.0));
    std::fill(mUpsamplingInputFilterHistory.begin(), mUpsamplingInputFilterHistory.end(), T(0.0));
    std::fill(mUpsamplingInputIIRState.begin(), mUpsamplingInputIIRState.end(), BiquadState {});
    std::fill(mMinPhaseDownIIRState.begin(), mMinPhaseDownIIRState.end(), BiquadState {});
    std::fill(mMinPhaseDownIIRPhase.begin(), mMinPhaseDownIIRPhase.end(), 0);
    std::fill(mCascadedHalfBandHistory.begin(), mCascadedHalfBandHistory.end(), T(0.0));
    std::fill(mCascadedHalfBandPhase.begin(), mCascadedHalfBandPhase.end(), 0);
    std::fill(mCascadedHalfBandInterpHistory.begin(), mCascadedHalfBandInterpHistory.end(), T(0.0));

    std::fill(mIIRInterpAState.begin(), mIIRInterpAState.end(), OnePoleAllpassState {});
    std::fill(mIIRInterpBState.begin(), mIIRInterpBState.end(), OnePoleAllpassState {});
    std::fill(mIIRPolyDecimEvenState.begin(), mIIRPolyDecimEvenState.end(), OnePoleAllpassState {});
    std::fill(mIIRPolyDecimOddState.begin(), mIIRPolyDecimOddState.end(), OnePoleAllpassState {});

    if (mResampler1 != nullptr)
    {
      mResampler1->ClearBuffer();
    }

    if (mResampler2 != nullptr)
    {
      mResampler2->ClearBuffer();
    }

    mDecimationPhase = 0;
    mOutputWritePos = 0;
  }

  // How big could the corresponding encapsulated buffer be for a buffer at the external sample rate of a given size?
  
int MaxEncapsulatedBlockSize(const int externalBlockSize) const
  {
    if (mUseCascadedHalfBandResampler)
      return externalBlockSize * std::max(1, mIntegerDownsampleFactor);

    return static_cast<int>(std::ceil(static_cast<double>(externalBlockSize) / mRatio1));
  }

  // Size of the multi-channel data for a given block size
  size_t DataSize(const int blockSize) const { return blockSize * NCHANS * sizeof(T); };

  void FallbackFunc(T** inputs, T** outputs, int n)
  {
    for (int i = 0; i < NCHANS; i++)
    {
      memcpy(outputs[i], inputs[i], n * sizeof(T));
    }
  }

  
void DesignAntiAliasFilter()
  {
    mAntiAliasEnabled = mRenderingSampleRate > (mInputSampleRate * 1.000001)
                        || mInputSampleRate > (mBandwidthSampleRate * 1.000001);

    mAntiAliasCoefficients.clear();
    mAntiAliasHistory.clear();
    mMinimumPhaseSections.clear();
    mMinimumPhaseState.clear();
    mDecimationFirCoefficients.clear();
    mDecimationFirHistory.clear();
    mUpsamplingInputFilterCoefficients.clear();
    mUpsamplingInputFilterHistory.clear();
    mHalfBandCoefficients.clear();
    mCascadedHalfBandHistory.clear();
    mCascadedHalfBandPhase.clear();
    mCascadedHalfBandBuffers.clear();
    mCascadedHalfBandInterpHistory.clear();
    mCascadedHalfBandInterpBuffers.clear();
    mCascadedHalfBandStages = 0;

    if (!mAntiAliasEnabled)
    {
      mDesignedFilterPhase = mFilterPhase;
      return;
    }

    if (mUseIntegerDownsampler && IsPowerOfTwo(mIntegerDownsampleFactor))
    {
      DesignCascadedHalfBandAntiAliasFilter();
      mDesignedFilterPhase = mFilterPhase;
      return;
    }

    // Internal fractional/non-power-of-two fallback.
    if (UsesMinimumPhaseCascadedFIR())
      DesignMinimumPhaseAntiAliasFilter();
    else
      DesignLinearPhaseAntiAliasFilter();

    DesignUpsamplingInputAntiImageFilter();
    if (UsesMinimumPhaseCascadedFIR())
      DesignMinimumPhaseIIRDownsampleFilter();

    mDesignedFilterPhase = mFilterPhase;
  }

  void DesignLinearPhaseAntiAliasFilter()
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr int numTaps = 255;
    const double effectiveInputSampleRate = std::min(mInputSampleRate, mBandwidthSampleRate);
    const double cutoff = std::min(0.49, 0.475 * effectiveInputSampleRate / mRenderingSampleRate);
    double sum = 0.0;

    mAntiAliasCoefficients.resize(numTaps);
    for (int n = 0; n < numTaps; n++)
    {
      const double x = n - 0.5 * (numTaps - 1);
      const double sinc = std::abs(x) < 1.0e-12 ? 2.0 * cutoff : std::sin(2.0 * pi * cutoff * x) / (pi * x);
      const double window = 0.42 - 0.5 * std::cos(2.0 * pi * n / (numTaps - 1))
                            + 0.08 * std::cos(4.0 * pi * n / (numTaps - 1));
      mAntiAliasCoefficients[n] = static_cast<T>(sinc * window);
      sum += mAntiAliasCoefficients[n];
    }

    if (std::abs(sum) > 1.0e-20)
      for (auto& c : mAntiAliasCoefficients)
        c = static_cast<T>(c / sum);

    mAntiAliasHistory.assign(NCHANS * (numTaps - 1), T(0.0));
  }

  void DesignMinimumPhaseFIRAntiAliasFilter()
  {
    DesignLinearPhaseAntiAliasFilter();
    ConvertFIRToMinimumPhase(mAntiAliasCoefficients);
    std::fill(mAntiAliasHistory.begin(), mAntiAliasHistory.end(), T(0.0));
  }

  std::vector<T> DesignKaiserLowpassFIR(int numTaps, double cutoff)
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr double beta = 10.5;
    const double denom = BesselI0(beta);
    double sum = 0.0;
    std::vector<T> coefficients(numTaps);

    for (int n = 0; n < numTaps; n++)
    {
      const double x = n - 0.5 * (numTaps - 1);
      const double sinc = std::abs(x) < 1.0e-12 ? 2.0 * cutoff : std::sin(2.0 * pi * cutoff * x) / (pi * x);
      const double r = (2.0 * n) / (numTaps - 1) - 1.0;
      const double window = BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / denom;
      coefficients[n] = static_cast<T>(sinc * window);
      sum += coefficients[n];
    }

    if (std::abs(sum) > 1.0e-20)
      for (auto& c : coefficients)
        c = static_cast<T>(c / sum);

    return coefficients;
  }

  void DesignDecimationFIRAntiAliasFilter(bool minimumPhase)
  {
    constexpr int tapsPerFactor = 128;
    constexpr double cutoffScale = 0.445;
    const double effectiveInputSampleRate = std::min(mInputSampleRate, mBandwidthSampleRate);
    const double effectiveFactor = std::max(1.0, mRenderingSampleRate / effectiveInputSampleRate);
    const int factor = std::max(1, static_cast<int>(std::ceil(effectiveFactor)));
    const int numTaps = tapsPerFactor * factor + 1;
    const double cutoff = std::min(0.49, cutoffScale / effectiveFactor);

    mDecimationFirCoefficients = DesignKaiserLowpassFIR(numTaps, cutoff);
    if (minimumPhase)
      ConvertFIRToMinimumPhase(mDecimationFirCoefficients);

    mDecimationFirHistory.assign(NCHANS * (numTaps - 1), T(0.0));
  }

  void ApplyAntiAliasFilter(size_t nFrames)
  {
    if (mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseCascadedFIR)
      ApplyMinimumPhaseAntiAliasFilter(nFrames);
    else
      ApplyLinearPhaseAntiAliasFilter(nFrames);
  }

  

  double GetStrictUpsamplingInputGuardCutoff() const
  {
    if (mInputSampleRate <= 0.0)
      return 0.49;

    // Same strict bandwidth idea as the down/up reconstruction filters:
    // host >= model -> model bandwidth
    // host <  model -> host bandwidth
    //
    // 0.455 * Fs gives ~21.84 kHz at 48 kHz, keeping the audible band flatter
    // while still preserving strict model/host bandwidth guarding before Nyquist.
    const double bandwidthBasis = GetStrictBandwidthBasisSampleRate();
    const double cutoffHz = 0.455 * bandwidthBasis;

    return std::min(0.49, std::max(0.001, cutoffHz / mInputSampleRate));
  }

  void DesignUpsamplingInputAntiImageFilter()
  {
    mUpsamplingInputFilterCoefficients.clear();
    mUpsamplingInputFilterHistory.clear();
    mUpsamplingInputIIRSections.clear();
    mUpsamplingInputIIRState.clear();

    // Linear Phase pre-upsample guard remains disabled:
    // the cascaded FIR up/down stages already provide the required interpolation
    // and downsample filtering.
    //
    // Minimum Phase gets a model-strict pre-upsample IIR guard only when the
    // host/output sample rate is higher than the model bandwidth basis, e.g.
    // host 96 kHz with a 48 kHz model. This uses the same cutoff/order policy
    // as the post-downsample/output model-strict guard.
    if (UsesMinimumPhaseCascadedFIR() && NeedsMinimumPhaseOutputModelStrictIIRGuard())
    {
      const int order = GetMinimumPhaseModelStrictIIROrder();
      const double cutoff = GetMinimumPhaseOutputModelStrictIIRCutoff();

      mUpsamplingInputIIRSections = DesignButterworthLowpass(cutoff, order);
      mUpsamplingInputIIRState.assign(
        static_cast<size_t>(NCHANS) * mUpsamplingInputIIRSections.size(), BiquadState {});
    }

    return;
  }






  T** PrepareUpsamplerInput(T** inputs, size_t nFrames)
  {
    if (!mUpsamplingInputIIRSections.empty())
    {
      const size_t numSections = mUpsamplingInputIIRSections.size();

      if (mUpsamplingInputIIRState.size() < static_cast<size_t>(NCHANS) * numSections)
        mUpsamplingInputIIRState.assign(static_cast<size_t>(NCHANS) * numSections, BiquadState {});

      for (int chan = 0; chan < NCHANS; chan++)
      {
        T* input = inputs[chan];
        T* output = mScratchExternalInputPointers.Get(chan);

        for (size_t s = 0; s < nFrames; s++)
        {
          T y = input[s];

          for (size_t section = 0; section < numSections; section++)
          {
            const auto& coeffs = mUpsamplingInputIIRSections[section];
            auto& state = mUpsamplingInputIIRState[static_cast<size_t>(chan) * numSections + section];

            const T out = coeffs.b0 * y + state.z1;
            state.z1 = coeffs.b1 * y - coeffs.a1 * out + state.z2;
            state.z2 = coeffs.b2 * y - coeffs.a2 * out;
            y = out;

            if (!std::isfinite(static_cast<double>(y)) || std::abs(static_cast<double>(y)) > 1.0e12)
            {
              for (size_t resetSection = 0; resetSection < numSections; resetSection++)
                mUpsamplingInputIIRState[static_cast<size_t>(chan) * numSections + resetSection] = {};
              y = T(0.0);
              break;
            }
          }

          output[s] = y;
        }
      }

      return mScratchExternalInputPointers.GetList();
    }

    if (mUpsamplingInputFilterCoefficients.empty())
      return inputs;

    const size_t numTaps = mUpsamplingInputFilterCoefficients.size();
    const size_t historyLen = numTaps - 1;

    if (mUpsamplingInputFilterHistory.size() < static_cast<size_t>(NCHANS) * historyLen)
      mUpsamplingInputFilterHistory.assign(static_cast<size_t>(NCHANS) * historyLen, T(0.0));

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* input = inputs[chan];
      T* output = mScratchExternalInputPointers.Get(chan);
      T* history = mUpsamplingInputFilterHistory.data() + static_cast<size_t>(chan) * historyLen;

      for (size_t s = 0; s < nFrames; s++)
      {
        T y = T(0.0);

        for (size_t k = 0; k < numTaps; k++)
        {
          const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
          const T x = inputIndex >= 0 ? input[static_cast<size_t>(inputIndex)]
                                      : history[historyLen + static_cast<size_t>(inputIndex)];
          y += mUpsamplingInputFilterCoefficients[k] * x;
        }

        output[s] = y;
      }

      if (nFrames >= historyLen)
      {
        std::copy(input + nFrames - historyLen, input + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input, input + nFrames, history + historyLen - nFrames);
      }
    }

    return mScratchExternalInputPointers.GetList();
  }

  
  void DesignMinimumPhaseIIRDownsampleFilter()
  {
    mMinPhaseDownIIRSections.clear();
    mMinPhaseDownIIRState.clear();
    mMinPhaseDownIIRPhase.clear();

    if (mInputSampleRate <= 0.0 || mRenderingSampleRate <= 0.0)
      return;

    int order = 16;
    const double cutoff = GetCascadedPrototypeCutoff();

    if (const char* envOrder = std::getenv("NAM_MINPHASE_DOWN_IIR_ORDER"))
    {
      const int parsed = std::atoi(envOrder);
      if (parsed >= 2)
        order = parsed;
    }

    if (order < 2)
      order = 2;
    if (order > 64)
      order = 64;
    if ((order & 1) != 0)
      order++;

    mMinPhaseDownIIRSections = DesignButterworthLowpass(cutoff, order);

    const size_t stages = static_cast<size_t>(std::max(0, mCascadedHalfBandStages));
    const size_t states = stages * static_cast<size_t>(NCHANS) * mMinPhaseDownIIRSections.size();

    mMinPhaseDownIIRState.assign(states, BiquadState {});
    mMinPhaseDownIIRPhase.assign(stages, 0);

    DesignMinimumPhaseOutputModelStrictIIRGuard();
  }

  bool NeedsMinimumPhaseOutputModelStrictIIRGuard() const
  {
    const double bandwidthBasis = GetStrictBandwidthBasisSampleRate();

    // Only needed when the host/output sample rate is higher than the model
    // bandwidth basis. Example: host 96 kHz, model 48 kHz.
    //
    // If host == model, the half-band decimator already ends at the same
    // Nyquist boundary and this extra output guard would only add roll-off.
    return mInputSampleRate > 0.0 && bandwidthBasis > 0.0 && mInputSampleRate > bandwidthBasis * 1.001;
  }

  double GetMinimumPhaseOutputModelStrictIIRCutoff() const
  {
    const double passband = GetStrictPassbandEdgeHz();
    const double stopband = GetStrictStopbandStartHz();

    // Tuned for host > model cases, e.g. 96 kHz host / 48 kHz model.
    //
    // With the default 40th-order Butterworth output guard, this bias gives:
    // - effectively unity gain at 20 kHz
    // - roughly the same attenuation at 24 kHz as Linear Phase (long)
    //
    // For a 48 kHz model bandwidth basis:
    // passband = 20 kHz, stopband = 24 kHz, cutoff ~= 21.69 kHz.
    double cutoffBias = 0.4224;

    if (const char* envBias = std::getenv("NAM_MINPHASE_OUTPUT_IIR_CUTOFF_BIAS"))
    {
      const double parsed = std::atof(envBias);
      if (parsed >= 0.0 && parsed <= 1.0)
        cutoffBias = parsed;
    }

    const double cutoffHz = passband + cutoffBias * std::max(0.0, stopband - passband);

    return std::min(0.49, std::max(0.001, cutoffHz / mInputSampleRate));
  }


  int GetMinimumPhaseModelStrictIIROrder() const
  {
    // Shared by the Minimum Phase pre-upsample and post-downsample/output
    // model-strict guards so both have the same response.
    //
    // 40th order keeps 20 kHz essentially flat while matching the Linear Phase
    // long attenuation around the 24 kHz model Nyquist boundary.
    int order = 40;

    if (const char* envOrder = std::getenv("NAM_MINPHASE_OUTPUT_IIR_ORDER"))
    {
      const int parsed = std::atoi(envOrder);
      if (parsed >= 2)
        order = parsed;
    }

    if (order < 2)
      order = 2;
    if (order > 64)
      order = 64;
    if ((order & 1) != 0)
      order++;

    return order;
  }

  void DesignMinimumPhaseOutputModelStrictIIRGuard()
  {
    mMinPhaseOutputStrictIIRSections.clear();
    mMinPhaseOutputStrictIIRState.clear();

    if (!NeedsMinimumPhaseOutputModelStrictIIRGuard())
      return;

    const int order = GetMinimumPhaseModelStrictIIROrder();
    const double cutoff = GetMinimumPhaseOutputModelStrictIIRCutoff();

    mMinPhaseOutputStrictIIRSections = DesignButterworthLowpass(cutoff, order);
    mMinPhaseOutputStrictIIRState.assign(
      static_cast<size_t>(NCHANS) * mMinPhaseOutputStrictIIRSections.size(), BiquadState {});
  }

  T ProcessMinimumPhaseOutputModelStrictIIRGuardSample(int chan, T x)
  {
    if (!NeedsMinimumPhaseOutputModelStrictIIRGuard())
      return x;

    if (mMinPhaseOutputStrictIIRSections.empty())
      DesignMinimumPhaseOutputModelStrictIIRGuard();

    if (mMinPhaseOutputStrictIIRSections.empty())
      return x;

    const size_t numSections = mMinPhaseOutputStrictIIRSections.size();
    const size_t requiredStateCount = static_cast<size_t>(NCHANS) * numSections;

    if (mMinPhaseOutputStrictIIRState.size() < requiredStateCount)
      mMinPhaseOutputStrictIIRState.assign(requiredStateCount, BiquadState {});

    const size_t base = static_cast<size_t>(chan) * numSections;

    T y = x;

    for (size_t section = 0; section < numSections; section++)
    {
      const auto& coeffs = mMinPhaseOutputStrictIIRSections[section];
      auto& state = mMinPhaseOutputStrictIIRState[base + section];

      const T out = coeffs.b0 * y + state.z1;
      state.z1 = coeffs.b1 * y - coeffs.a1 * out + state.z2;
      state.z2 = coeffs.b2 * y - coeffs.a2 * out;
      y = out;

      if (!std::isfinite(static_cast<double>(y)) || std::abs(static_cast<double>(y)) > 1.0e12)
      {
        for (size_t resetSection = 0; resetSection < numSections; resetSection++)
          mMinPhaseOutputStrictIIRState[base + resetSection] = {};

        y = T(0.0);
        break;
      }
    }

    return y;
  }


T** PrepareDownsamplerInput(size_t nFrames)
  {
    if (!mAntiAliasEnabled)
      return mEncapsulatedOutputPointers.GetList();

    if (mUseIntegerDownsampler && UsesCascadedFIR())
      return mEncapsulatedOutputPointers.GetList();

    ApplyAntiAliasFilter(nFrames);
    return mAntiAliasOutputPointers.GetList();
  }



  void ApplyLinearPhaseAntiAliasFilter(size_t nFrames)
  {
    const size_t numTaps = mAntiAliasCoefficients.size();
    const size_t historyLen = numTaps - 1;

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* input = mEncapsulatedOutputPointers.Get(chan);
      T* output = mAntiAliasOutputPointers.Get(chan);
      T* history = mAntiAliasHistory.data() + chan * historyLen;

      for (size_t s = 0; s < nFrames; s++)
      {
        T y = T(0.0);
        for (size_t k = 0; k < numTaps; k++)
        {
          const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
          const T x = inputIndex >= 0 ? input[inputIndex] : history[historyLen + inputIndex];
          y += mAntiAliasCoefficients[k] * x;
        }
        output[s] = y;
      }

      if (nFrames >= historyLen)
      {
        std::copy(input + nFrames - historyLen, input + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input, input + nFrames, history + historyLen - nFrames);
      }
    }
  }

  struct BiquadCoefficients
  {
    T b0 = T(1.0);
    T b1 = T(0.0);
    T b2 = T(0.0);
    T a1 = T(0.0);
    T a2 = T(0.0);
  };

  struct BiquadState
  {
    T z1 = T(0.0);
    T z2 = T(0.0);
  };

  static double BesselI0(double x)
  {
    double sum = 1.0;
    double term = 1.0;
    const double y = 0.25 * x * x;
    for (int k = 1; k < 32; k++)
    {
      term *= y / static_cast<double>(k * k);
      sum += term;
      if (term < 1.0e-12 * sum)
        break;
    }
    return sum;
  }

  static void FFT(std::vector<std::complex<double>>& data, bool inverse)
  {
    const size_t n = data.size();
    for (size_t i = 1, j = 0; i < n; i++)
    {
      size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j)
        std::swap(data[i], data[j]);
    }

    constexpr double pi = 3.14159265358979323846264338327950288;
    for (size_t len = 2; len <= n; len <<= 1)
    {
      const double angle = (inverse ? 2.0 : -2.0) * pi / static_cast<double>(len);
      const std::complex<double> wLen(std::cos(angle), std::sin(angle));
      for (size_t i = 0; i < n; i += len)
      {
        std::complex<double> w(1.0, 0.0);
        for (size_t j = 0; j < len / 2; j++)
        {
          const auto u = data[i + j];
          const auto v = data[i + j + len / 2] * w;
          data[i + j] = u + v;
          data[i + j + len / 2] = u - v;
          w *= wLen;
        }
      }
    }

    if (inverse)
      for (auto& x : data)
        x /= static_cast<double>(n);
  }

  void ConvertFIRToMinimumPhase(std::vector<T>& coefficients)
  {
    const size_t numTaps = coefficients.size();
    size_t fftSize = 1;
    while (fftSize < 16 * numTaps)
      fftSize <<= 1;

    std::vector<std::complex<double>> spectrum(fftSize, std::complex<double>{0.0, 0.0});
    for (size_t i = 0; i < numTaps; i++)
      spectrum[i] = static_cast<double>(coefficients[i]);

    FFT(spectrum, false);
    for (auto& x : spectrum)
      x = std::log(std::max(1.0e-14, std::abs(x)));

    FFT(spectrum, true);
    std::vector<std::complex<double>> minimumCepstrum(fftSize, std::complex<double>{0.0, 0.0});
    minimumCepstrum[0] = spectrum[0].real();
    for (size_t i = 1; i < fftSize / 2; i++)
      minimumCepstrum[i] = 2.0 * spectrum[i].real();
    if ((fftSize & 1U) == 0U)
      minimumCepstrum[fftSize / 2] = spectrum[fftSize / 2].real();

    FFT(minimumCepstrum, false);
    for (auto& x : minimumCepstrum)
      x = std::exp(x);

    FFT(minimumCepstrum, true);
    double sum = 0.0;
    for (size_t i = 0; i < numTaps; i++)
    {
      coefficients[i] = static_cast<T>(minimumCepstrum[i].real());
      sum += coefficients[i];
    }

    if (std::abs(sum) > 1.0e-20)
      for (auto& c : coefficients)
        c = static_cast<T>(c / sum);
  }

  std::vector<BiquadCoefficients> DesignButterworthLowpass(double cutoff, int order)
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    cutoff = std::min(0.49, std::max(0.001, cutoff));
    const int numSections = order / 2;
    const double omega = 2.0 * pi * cutoff;
    const double sinOmega = std::sin(omega);
    const double cosOmega = std::cos(omega);

    std::vector<BiquadCoefficients> sections(numSections);
    for (int section = 0; section < numSections; section++)
    {
      const double q = 1.0 / (2.0 * std::cos((2.0 * section + 1.0) * pi / (2.0 * order)));
      const double alpha = sinOmega / (2.0 * q);
      const double a0 = 1.0 + alpha;

      BiquadCoefficients coeffs;
      coeffs.b0 = static_cast<T>((1.0 - cosOmega) * 0.5 / a0);
      coeffs.b1 = static_cast<T>((1.0 - cosOmega) / a0);
      coeffs.b2 = static_cast<T>((1.0 - cosOmega) * 0.5 / a0);
      coeffs.a1 = static_cast<T>((-2.0 * cosOmega) / a0);
      coeffs.a2 = static_cast<T>((1.0 - alpha) / a0);
      sections[section] = coeffs;
    }
    return sections;
  }

  void DesignMinimumPhaseAntiAliasFilter()
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr int order = 32;
    constexpr int numSections = order / 2;
    constexpr double rippleDb = 0.1;
    constexpr double cutoffScale = 0.445;
    const double effectiveInputSampleRate = std::min(mInputSampleRate, mBandwidthSampleRate);
    const double factor = mRenderingSampleRate / effectiveInputSampleRate;
    const double cutoff = std::min(0.49, cutoffScale / std::max(1.0, factor));
    const double epsilon = std::sqrt(std::pow(10.0, rippleDb / 10.0) - 1.0);
    const double mu = std::asinh(1.0 / epsilon) / static_cast<double>(order);
    const double warpedCutoff = 2.0 * std::tan(pi * cutoff);

    mMinimumPhaseSections.resize(numSections);
    for (int section = 0; section < numSections; section++)
    {
      const double theta = pi * (2.0 * section + 1.0) / (2.0 * order);
      const double sigma = -std::sinh(mu) * std::sin(theta) * warpedCutoff;
      const double omega = std::cosh(mu) * std::cos(theta) * warpedCutoff;
      const double b = -2.0 * sigma;
      const double c = sigma * sigma + omega * omega;
      const double a0 = 4.0 + 2.0 * b + c;

      BiquadCoefficients coeffs;
      coeffs.b0 = static_cast<T>(c / a0);
      coeffs.b1 = static_cast<T>((2.0 * c) / a0);
      coeffs.b2 = static_cast<T>(c / a0);
      coeffs.a1 = static_cast<T>((-8.0 + 2.0 * c) / a0);
      coeffs.a2 = static_cast<T>((4.0 - 2.0 * b + c) / a0);
      mMinimumPhaseSections[section] = coeffs;
    }

    mMinimumPhaseState.assign(NCHANS * mMinimumPhaseSections.size(), BiquadState {});
  }

  void ApplyMinimumPhaseAntiAliasFilter(size_t nFrames)
  {
    ApplyMinimumPhaseFilter(mEncapsulatedOutputPointers.GetList(), mAntiAliasOutputPointers.GetList(), nFrames,
                            mMinimumPhaseState);
  }

  void ApplyMinimumPhaseFilter(T** input, T** output, size_t nFrames, std::vector<BiquadState>& state)
  {
    for (int chan = 0; chan < NCHANS; chan++)
    {
      for (size_t s = 0; s < nFrames; s++)
      {
        T y = input[chan][s];
        for (size_t section = 0; section < mMinimumPhaseSections.size(); section++)
        {
          const auto& coeffs = mMinimumPhaseSections[section];
          auto& sectionState = state[chan * mMinimumPhaseSections.size() + section];
          const T out = coeffs.b0 * y + sectionState.z1;
          sectionState.z1 = coeffs.b1 * y - coeffs.a1 * out + sectionState.z2;
          sectionState.z2 = coeffs.b2 * y - coeffs.a2 * out;
          y = out;

          if (!std::isfinite(static_cast<double>(y)) || std::abs(static_cast<double>(y)) > 1.0e12)
          {
            for (size_t resetSection = 0; resetSection < mMinimumPhaseSections.size(); resetSection++)
              state[chan * mMinimumPhaseSections.size() + resetSection] = {};
            y = T(0.0);
            break;
          }
        }
        output[chan][s] = y;
      }
    }
  }

  void ApplyFIR(T** input, T** output, size_t nFrames, const std::vector<T>& coefficients, std::vector<T>& historyBuffer)
  {
    const size_t numTaps = coefficients.size();
    const size_t historyLen = numTaps - 1;

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = historyBuffer.data() + chan * historyLen;

      for (size_t s = 0; s < nFrames; s++)
      {
        T y = T(0.0);
        for (size_t k = 0; k < numTaps; k++)
        {
          const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
          const T x = inputIndex >= 0 ? input[chan][inputIndex] : history[historyLen + inputIndex];
          y += coefficients[k] * x;
        }
        output[chan][s] = y;
      }

      if (nFrames >= historyLen)
      {
        std::copy(input[chan] + nFrames - historyLen, input[chan] + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input[chan], input[chan] + nFrames, history + historyLen - nFrames);
      }
    }
  }

  
static bool IsPowerOfTwo(int x)
  {
    return x > 0 && (x & (x - 1)) == 0;
  }

  static int Log2PowerOfTwo(int x)
  {
    int result = 0;
    while (x > 1)
    {
      x >>= 1;
      result++;
    }
    return result;
  }

  std::vector<T> DesignHalfBandLowpassFIR(int numTaps)
  {
    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr double cutoff = 0.25; // Half-band cutoff, normalized to the stage sample rate.
    constexpr double beta = 10.5;   // Kaiser window, roughly high stop-band attenuation.
    const double denom = BesselI0(beta);

    if ((numTaps % 2) == 0)
      numTaps++;

    const int center = numTaps / 2;
    std::vector<T> coefficients(numTaps, T(0.0));
    double sum = 0.0;

    for (int n = 0; n < numTaps; n++)
    {
      const int m = n - center;
      double h = 0.0;

      if (m == 0)
      {
        h = 2.0 * cutoff;
      }
      else if ((m & 1) != 0)
      {
        h = std::sin(2.0 * pi * cutoff * static_cast<double>(m)) / (pi * static_cast<double>(m));
      }
      else
      {
        // Exact half-band zero taps, except the center tap.
        h = 0.0;
      }

      const double r = (2.0 * n) / (numTaps - 1) - 1.0;
      const double window = BesselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / denom;

      coefficients[n] = static_cast<T>(h * window);
      sum += static_cast<double>(coefficients[n]);
    }

    if (std::abs(sum) > 1.0e-20)
      for (auto& c : coefficients)
        c = static_cast<T>(static_cast<double>(c) / sum);

    return coefficients;
  }

  // BEGIN NAM_OS_THREE_STRICT_BANDWIDTH_MODES
bool UsesMinimumPhaseCascadedFIR() const
  {
    return mFilterPhase == EAntiAliasFilterPhase::MinimumPhaseCascadedFIR;
  }

bool UsesLinearPhaseCascadedFIR() const
  {
    return mFilterPhase == EAntiAliasFilterPhase::LinearCascadedFIRShort
           || mFilterPhase == EAntiAliasFilterPhase::LinearCascadedFIRLong;
  }

bool UsesCascadedFIR() const
  {
    return UsesMinimumPhaseCascadedFIR() || UsesLinearPhaseCascadedFIR();
  }

  double GetStrictBandwidthBasisSampleRate() const
  {
    // Restored model-bandwidth-strict rule:
    //   host >= model -> use model bandwidth
    //   host <  model -> use host bandwidth
    return std::min(mInputSampleRate, mBandwidthSampleRate);
  }

  double GetStrictPassbandEdgeHz() const
  {
    const double bandwidthBasis = GetStrictBandwidthBasisSampleRate();
    const double nyquist = 0.5 * bandwidthBasis;

    // 48 kHz basis -> 20 kHz passband.
    // 96 kHz basis -> 40 kHz passband.
    // 44.1 kHz basis -> still try to keep 20 kHz if Nyquist allows it.
    const double scaledPassband = (20000.0 / 24000.0) * nyquist;
    const double desiredPassband = std::max(20000.0, scaledPassband);

    // Leave a small transition band below the selected Nyquist.
    return std::min(desiredPassband, 0.95 * nyquist);
  }

  double GetStrictStopbandStartHz() const
  {
    return 0.5 * GetStrictBandwidthBasisSampleRate();
  }

  double GetCascadedPrototypeCutoff() const
  {
    const double passband = GetStrictPassbandEdgeHz();
    const double stopband = GetStrictStopbandStartHz();

    // Bias the cutoff closer to the stopband than the midpoint. This keeps the
    // response effectively unity through the intended passband, especially at 20 kHz,
    // while the long mode still provides strong rejection near the strict Nyquist.
    const double cutoffHz = passband + 0.75 * std::max(0.0, stopband - passband);

    // The prototype filter is applied in each 2x stage. The final 2x stage runs
    // at 2 * hostSampleRate, so this is the normalized cutoff for that stage.
    const double finalStageSampleRate = 2.0 * mInputSampleRate;
    return std::min(0.49, std::max(0.001, cutoffHz / finalStageSampleRate));
  }

int GetCascadedPrototypeNumTaps() const
  {
    return mFilterPhase == EAntiAliasFilterPhase::LinearCascadedFIRLong ? 513 : 129;
  }

  void SelectCascadedFIRCoefficients()
  {
    const int numTaps = GetCascadedPrototypeNumTaps();
    const double cutoff = GetCascadedPrototypeCutoff();

    mHalfBandCoefficients = DesignKaiserLowpassFIR(numTaps, cutoff);

    if (UsesMinimumPhaseCascadedFIR())
      ConvertFIRToMinimumPhase(mHalfBandCoefficients);

    // Enforce exact unity DC gain in the plugin's sample type.
    T dcGain = T(0.0);

    for (const auto& coeff : mHalfBandCoefficients)
      dcGain += coeff;

    if (std::abs(static_cast<double>(dcGain)) > 1.0e-20)
    {
      for (auto& coeff : mHalfBandCoefficients)
        coeff /= dcGain;
    }

    // Reversed copy for forward-access SIMD in the decimation steady-state loop.
    mHalfBandCoefficientsReversed.assign(mHalfBandCoefficients.rbegin(), mHalfBandCoefficients.rend());
  }
  // END NAM_OS_THREE_STRICT_BANDWIDTH_MODES

  
  void BuildHalfBandPolyphaseTapCache()
  {
    for (size_t parity = 0; parity < 2; parity++)
    {
      mHalfBandPolyphaseTapIndices[parity].clear();
      mHalfBandPolyphaseTapCoefficients[parity].clear();
    }

    for (size_t k = 0; k < mHalfBandCoefficients.size(); k++)
    {
      const size_t parity = k & 1U;
      mHalfBandPolyphaseTapIndices[parity].push_back(k);
      mHalfBandPolyphaseTapCoefficients[parity].push_back(mHalfBandCoefficients[k]);
    }
  }

void DesignCascadedHalfBandAntiAliasFilter()
  {
    mCascadedHalfBandStages = 0;

    int factor = mIntegerDownsampleFactor;

    while (factor > 1)
    {
      mCascadedHalfBandStages++;
      factor >>= 1;
    }

    SelectCascadedFIRCoefficients();
    BuildHalfBandPolyphaseTapCache();

    if (mCascadedHalfBandStages <= 0 || mHalfBandCoefficients.empty())
      return;

    const size_t historyLen = mHalfBandCoefficients.size() - 1;

    mCascadedHalfBandHistory.assign(
      static_cast<size_t>(mCascadedHalfBandStages) * NCHANS * historyLen,
      T(0.0));

    mCascadedHalfBandPhase.assign(
      static_cast<size_t>(mCascadedHalfBandStages) * NCHANS,
      0);

    mCascadedHalfBandBuffers.resize(static_cast<size_t>(mCascadedHalfBandStages));

    mCascadedHalfBandInterpHistory.assign(
      static_cast<size_t>(mCascadedHalfBandStages) * NCHANS * historyLen,
      T(0.0));

    mCascadedHalfBandInterpBuffers.resize(static_cast<size_t>(mCascadedHalfBandStages));
    mIIRHalfBandBuffers.resize(static_cast<size_t>(mCascadedHalfBandStages));
    EnsureIIRHalfBandState();

    // Critical anti-image guard before upsampling:
    // keep ultrasonic/images out of the NAM model input. Without this, the
    // non-linear model can fold/intermodulate upsampling images back into
    // audible band, which then survives even a correct downsampler.
    DesignUpsamplingInputAntiImageFilter();
  }

  






double GetCascadedHalfBandOneWayLatencySamples() const
  {
    if (UsesMinimumPhaseCascadedFIR())
      return 0.0;

    if (mCascadedHalfBandStages <= 0 || mHalfBandCoefficients.empty())
      return 0.0;

    const double groupDelay = 0.5 * static_cast<double>(mHalfBandCoefficients.size() - 1);
    double latencyAtExternalRate = 0.0;

    for (int stage = 0; stage < mCascadedHalfBandStages; stage++)
    {
      const int divisorToExternalRate = 1 << (mCascadedHalfBandStages - stage);
      latencyAtExternalRate += groupDelay / static_cast<double>(divisorToExternalRate);
    }

    return latencyAtExternalRate;
  }

  int GetCascadedHalfBandLatency() const
  {
    return static_cast<int>(std::ceil(GetCascadedHalfBandOneWayLatencySamples()));
  }

int GetCascadedHalfBandRoundTripLatency() const
  {
    if (UsesMinimumPhaseCascadedFIR())
      return 0;

    if (mCascadedHalfBandStages <= 0 || mHalfBandCoefficients.empty())
      return 0;

    return static_cast<int>(std::ceil(2.0 * GetCascadedHalfBandOneWayLatencySamples()));
  }

  size_t HalfBandInterpolateBy2Stage(int stage, T** input, size_t nFrames, T** output, size_t maxOutputFrames)
  {
    // Exact FIR/polyphase interpolation by 2 with cached tap lists and
    // prefix/steady-state split.
    //
    // First historyLen output samples may read the previous block history.
    // After that, every tap reads the current input block, so the hot loop has
    // no per-tap history branch. Coefficients/cutoff/phase are unchanged.
    const size_t historyLen = mHalfBandCoefficients.size() - 1;
    const size_t stageHistoryOffset = static_cast<size_t>(stage) * NCHANS * historyLen;
    const size_t outputFrames = std::min(maxOutputFrames, nFrames * 2);
    const size_t steadyStart = std::min(outputFrames, historyLen);

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = mCascadedHalfBandInterpHistory.data() + stageHistoryOffset + chan * historyLen;
      T* in = input[chan];
      T* out = output[chan];

      // Prefix: preserve exact increasing-k accumulation order while handling history.
      for (size_t os = 0; os < steadyStart; os++)
      {
        T y = T(0.0);

        const size_t parity = os & 1U;
        const auto& tapIndices = mHalfBandPolyphaseTapIndices[parity];
        const auto& tapCoeffs = mHalfBandPolyphaseTapCoefficients[parity];

        size_t ti = 0;
        const size_t n = tapIndices.size();

        // k <= os reads current input.
        for (; ti < n; ti++)
        {
          const size_t k = tapIndices[ti];
          if (k > os)
            break;

          y += tapCoeffs[ti] * in[(os - k) >> 1];
        }

        // k > os reads zero-stuffed history.
        for (; ti < n; ti++)
        {
          const size_t k = tapIndices[ti];
          const long zIndex = static_cast<long>(os) - static_cast<long>(k);
          y += tapCoeffs[ti] * history[historyLen + zIndex];
        }

        out[os] = T(2.0) * y;
      }

      // Steady state: all tap reads are in the current input block.
      for (size_t os = steadyStart; os < outputFrames; os++)
      {
        T y = T(0.0);

        const size_t parity = os & 1U;
        const auto& tapIndices = mHalfBandPolyphaseTapIndices[parity];
        const auto& tapCoeffs = mHalfBandPolyphaseTapCoefficients[parity];

        size_t ti = 0;
        const size_t n = tapIndices.size();

        for (; ti + 3 < n; ti += 4)
        {
          const size_t k0 = tapIndices[ti + 0];
          const size_t k1 = tapIndices[ti + 1];
          const size_t k2 = tapIndices[ti + 2];
          const size_t k3 = tapIndices[ti + 3];

          y += tapCoeffs[ti + 0] * in[(os - k0) >> 1];
          y += tapCoeffs[ti + 1] * in[(os - k1) >> 1];
          y += tapCoeffs[ti + 2] * in[(os - k2) >> 1];
          y += tapCoeffs[ti + 3] * in[(os - k3) >> 1];
        }

        for (; ti < n; ti++)
        {
          const size_t k = tapIndices[ti];
          y += tapCoeffs[ti] * in[(os - k) >> 1];
        }

        out[os] = T(2.0) * y;
      }

      // Keep the same zero-stuffed history representation as the legacy path.
      if (outputFrames >= historyLen)
      {
        for (size_t i = 0; i < historyLen; i++)
        {
          const size_t zIndex = outputFrames - historyLen + i;
          history[i] = ((zIndex & 1U) == 0U) ? in[zIndex >> 1] : T(0.0);
        }
      }
      else
      {
        std::move(history + outputFrames, history + historyLen, history);

        for (size_t i = 0; i < outputFrames; i++)
        {
          const size_t zIndex = i;
          history[historyLen - outputFrames + i] =
            ((zIndex & 1U) == 0U) ? in[zIndex >> 1] : T(0.0);
        }
      }
    }

    return outputFrames;
  }





  size_t CascadedHalfBandUpsampleBlock(T** input, size_t nFrames, T** output, size_t maxOutputFrames)
  {
    T** stageInput = input;
    size_t stageInputFrames = nFrames;

    std::vector<T*> previousStagePointers(static_cast<size_t>(NCHANS));

    for (int stage = 0; stage < mCascadedHalfBandStages; stage++)
    {
      const bool finalStage = stage == (mCascadedHalfBandStages - 1);
      const size_t maxStageOutputFrames = finalStage ? maxOutputFrames : stageInputFrames * 2;

      if (finalStage)
      {
        return HalfBandInterpolateBy2Stage(stage, stageInput, stageInputFrames, output, maxStageOutputFrames);
      }

      auto& buffer = mCascadedHalfBandInterpBuffers[static_cast<size_t>(stage)];
      const size_t requiredBufferSize = static_cast<size_t>(NCHANS) * maxStageOutputFrames;
      if (buffer.size() < requiredBufferSize)
        buffer.resize(requiredBufferSize);

      std::vector<T*> stageOutputPointers(static_cast<size_t>(NCHANS));

      for (int chan = 0; chan < NCHANS; chan++)
        stageOutputPointers[static_cast<size_t>(chan)] = buffer.data() + chan * maxStageOutputFrames;

      const size_t produced =
        HalfBandInterpolateBy2Stage(stage, stageInput, stageInputFrames, stageOutputPointers.data(), maxStageOutputFrames);

      previousStagePointers = std::move(stageOutputPointers);
      stageInput = previousStagePointers.data();
      stageInputFrames = produced;
    }

    return 0;
  }











  size_t IIRStateIndex(int stage, int chan, int section) const
  {
    return (static_cast<size_t>(stage) * NCHANS + static_cast<size_t>(chan)) * kIIRHalfBandSections
           + static_cast<size_t>(section);
  }


  static T ProcessOnePoleAllpass(T x, T a, OnePoleAllpassState& s)
  {
    // HIIR-style first-order all-pass section:
    //
    //   H(z) = (a + z^-1) / (1 + a z^-1)
    //
    // Difference equation:
    //
    //   y[n] = a*x[n] + x[n-1] - a*y[n-1]
    //
    // The coefficient table used by IIRHalfBandCoeffA/B is a first-order
    // all-pass half-band table. Treating these coefficients as second-order
    // sections breaks the half-band response and makes Minimum Phase behave
    // like a bad/near-1x anti-alias path.
    const T y = a * x + s.x1 - a * s.y1;

    s.x1 = x;
    s.y1 = y;

    return y;
  }

  static T IIRHalfBandCoeffA(int i)
  {
    static constexpr double coeffs[kIIRHalfBandSections] = {
      0.036681502163648017,
      0.2746317593794541,
      0.56109896978791948,
      0.769741833862266,
      0.8922608180038789,
      0.962094548378084
    };
    return static_cast<T>(coeffs[i]);
  }
  static T IIRHalfBandCoeffB(int i)
  {
    static constexpr double coeffs[kIIRHalfBandSections] = {
      0.13654762463195771,
      0.42313861743656667,
      0.6775400499741616,
      0.839889624849638,
      0.9315419599631839,
      0.9878163707328971
    };
    return static_cast<T>(coeffs[i]);
  }


  void EnsureIIRHalfBandState()
  {
    const size_t stateCount = static_cast<size_t>(std::max(0, mCascadedHalfBandStages)) * NCHANS * kIIRHalfBandSections;

    mIIRInterpAState.assign(stateCount, OnePoleAllpassState {});
    mIIRInterpBState.assign(stateCount, OnePoleAllpassState {});
  }



  bool UseMinimumPhaseIIRCleanFusedFallback() const
  {
    // TRUE_POLYPHASE is now the default Minimum Phase downsampler.
    //
    // Set NAM_MINPHASE_IIR_CLEAN_FUSED=1 only for A/B regression testing
    // against the previous verified fused Butterworth IIR decimator.
    const char* v = std::getenv("NAM_MINPHASE_IIR_CLEAN_FUSED");
    return v != nullptr && v[0] != '\0' && v[0] != '0' && v[0] != 'f' && v[0] != 'F'
           && v[0] != 'n' && v[0] != 'N';
  }

  void EnsureIIRPolyphaseDecimatorState()
  {
    const size_t stateCount = static_cast<size_t>(std::max(0, mCascadedHalfBandStages)) * NCHANS * kIIRHalfBandSections;

    mIIRPolyDecimEvenState.assign(stateCount, OnePoleAllpassState {});
    mIIRPolyDecimOddState.assign(stateCount, OnePoleAllpassState {});
  }

  T ProcessIIRHalfBandPolyphaseDecimEvenBranch(int stage, int chan, T x)
  {
    // Correct low-pass analysis orientation:
    //   even input phase -> B all-pass branch
    //   odd  input phase -> A all-pass branch
    T y = x;
    for (int i = 0; i < kIIRHalfBandSections; i++)
      y = ProcessOnePoleAllpass(y, IIRHalfBandCoeffB(i), mIIRPolyDecimEvenState[IIRStateIndex(stage, chan, i)]);

    return y;
  }

  T ProcessIIRHalfBandPolyphaseDecimOddBranch(int stage, int chan, T x)
  {
    T y = x;
    for (int i = 0; i < kIIRHalfBandSections; i++)
      y = ProcessOnePoleAllpass(y, IIRHalfBandCoeffA(i), mIIRPolyDecimOddState[IIRStateIndex(stage, chan, i)]);

    return y;
  }

  T ProcessIIRHalfBandInterpBranchA(int stage, int chan, T x)
  {
    T y = x;
    for (int i = 0; i < kIIRHalfBandSections; i++)
      y = ProcessOnePoleAllpass(y, IIRHalfBandCoeffA(i), mIIRInterpAState[IIRStateIndex(stage, chan, i)]);

    return y;
  }

  T ProcessIIRHalfBandInterpBranchB(int stage, int chan, T x)
  {
    T y = x;
    for (int i = 0; i < kIIRHalfBandSections; i++)
      y = ProcessOnePoleAllpass(y, IIRHalfBandCoeffB(i), mIIRInterpBState[IIRStateIndex(stage, chan, i)]);

    return y;
  }
  size_t IIRHalfBandUpsampleBy2Stage(int stage, T** input, size_t nFrames, T** output, size_t maxOutputFrames)
  {
    // Correct all-pass/polyphase interpolation:
    //   y[2n]   = A(input[n])
    //   y[2n+1] = B(input[n])
    //
    // The previous implementation drove a recursive all-pass network with
    // zero-stuffed high-rate samples. That is not the polyphase form and it
    // badly changes the passband/stopband behavior.
    const size_t outputFrames = std::min(maxOutputFrames, nFrames * 2);

    for (int chan = 0; chan < NCHANS; chan++)
    {
      for (size_t n = 0; n < nFrames; n++)
      {
        const size_t evenIndex = n * 2;
        const size_t oddIndex = evenIndex + 1;

        const T x = input[chan][n];
        const T even = ProcessIIRHalfBandInterpBranchA(stage, chan, x);
        const T odd = ProcessIIRHalfBandInterpBranchB(stage, chan, x);

        if (evenIndex < outputFrames)
          output[chan][evenIndex] = even;
        if (oddIndex < outputFrames)
          output[chan][oddIndex] = odd;
      }
    }

    return outputFrames;
  }



  size_t CascadedIIRHalfBandUpsampleBlock(T** input, size_t nFrames, T** output, size_t maxOutputFrames)
  {
    T** stageInput = input;
    size_t stageInputFrames = nFrames;
    std::array<T*, NCHANS> previousStagePointers {};

    for (int stage = 0; stage < mCascadedHalfBandStages; stage++)
    {
      const bool finalStage = stage == (mCascadedHalfBandStages - 1);
      const size_t maxStageOutputFrames = finalStage ? maxOutputFrames : stageInputFrames * 2;

      if (finalStage)
        return IIRHalfBandUpsampleBy2Stage(stage, stageInput, stageInputFrames, output, maxStageOutputFrames);

      auto& buffer = mIIRHalfBandBuffers[static_cast<size_t>(stage)];
      const size_t requiredBufferSize = static_cast<size_t>(NCHANS) * maxStageOutputFrames;
      if (buffer.size() < requiredBufferSize)
        buffer.resize(requiredBufferSize);

      std::array<T*, NCHANS> stageOutputPointers {};
      for (int chan = 0; chan < NCHANS; chan++)
        stageOutputPointers[static_cast<size_t>(chan)] =
          buffer.data() + static_cast<size_t>(chan) * maxStageOutputFrames;

      const size_t produced = IIRHalfBandUpsampleBy2Stage(
        stage, stageInput, stageInputFrames, stageOutputPointers.data(), maxStageOutputFrames);

      previousStagePointers = stageOutputPointers;
      stageInput = previousStagePointers.data();
      stageInputFrames = produced;
    }

    return 0;
  }














  void ProcessBlockRealtimeIIRHalfBand(T** inputs, T** outputs, int nFrames, BlockProcessFunc func)
  {
    const bool profile =
      []()
      {
        const char* v = std::getenv("NAM_RESAMPLER_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0' && v[0] != 'f' && v[0] != 'F'
               && v[0] != 'n' && v[0] != 'N';
      }();

    const bool cleanFusedFallback = UseMinimumPhaseIIRCleanFusedFallback();
    const bool truePolyphase = !cleanFusedFallback;
    const char* modeLabel =
      truePolyphase ? "MinimumPhaseIIR_TRUE_POLYPHASE" : "MinimumPhaseIIR_CLEAN_FUSED_FALLBACK";

    using ProfileClock = std::chrono::high_resolution_clock;
    const auto t0 = profile ? ProfileClock::now() : ProfileClock::time_point {};

    if (mIIRInterpAState.empty() || mIIRInterpBState.empty())
      EnsureIIRHalfBandState();

    if (truePolyphase && (mIIRPolyDecimEvenState.empty() || mIIRPolyDecimOddState.empty()))
      EnsureIIRPolyphaseDecimatorState();

    if (mUpsamplingInputIIRSections.empty())
      DesignUpsamplingInputAntiImageFilter();

    mOutputWritePos = 0;

    const auto tUp0 = profile ? ProfileClock::now() : ProfileClock::time_point {};

    const size_t upsampledFrames =
      CascadedIIRHalfBandUpsampleBlock(PrepareUpsamplerInput(inputs, static_cast<size_t>(nFrames)),
                                       static_cast<size_t>(nFrames),
                                       mEncapsulatedInputPointers.GetList(),
                                       static_cast<size_t>(mMaxEncapsulatedBlockSize));

    const auto tUp1 = profile ? ProfileClock::now() : ProfileClock::time_point {};

    if (upsampledFrames > static_cast<size_t>(mMaxEncapsulatedBlockSize))
      throw std::runtime_error("Realtime IIR upsampler produced more samples than the model buffer can hold.");

    const auto tModel0 = profile ? ProfileClock::now() : ProfileClock::time_point {};

    func(mEncapsulatedInputPointers.GetList(),
         mEncapsulatedOutputPointers.GetList(),
         static_cast<int>(upsampledFrames));

    const auto tModel1 = profile ? ProfileClock::now() : ProfileClock::time_point {};
    const auto tDown0 = profile ? ProfileClock::now() : ProfileClock::time_point {};

    if (truePolyphase)
      CascadedMinimumPhaseIIRTruePolyphaseDecimateBlock(mEncapsulatedOutputPointers.GetList(), upsampledFrames, outputs, nFrames);
    else
      CascadedMinimumPhaseIIRFusedDecimateBlock(mEncapsulatedOutputPointers.GetList(), upsampledFrames, outputs, nFrames);

    const auto tDown1 = profile ? ProfileClock::now() : ProfileClock::time_point {};

    const auto populated = mOutputWritePos;
    if (populated < static_cast<size_t>(nFrames))
    {
      for (int c = 0; c < NCHANS; c++)
      {
        const T lastSample = populated > 0 ? outputs[c][populated - 1] : T(0.0);
        for (int i = static_cast<int>(populated); i < nFrames; i++)
          outputs[c][i] = lastSample;
      }
    }

    if (profile)
    {
      const auto t1 = ProfileClock::now();

      auto ms = [](ProfileClock::time_point a, ProfileClock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };

      static std::atomic<unsigned long long> sProfileBlocks {0};
      static std::mutex sProfileMutex;
      static double sTotalMs = 0.0;
      static double sUpMs = 0.0;
      static double sModelMs = 0.0;
      static double sDownMs = 0.0;

      const double upMs = ms(tUp0, tUp1);
      const double modelMs = ms(tModel0, tModel1);
      const double downMs = ms(tDown0, tDown1);
      const double totalMs = ms(t0, t1);
      const auto blocks = sProfileBlocks.fetch_add(1) + 1;

      {
        std::lock_guard<std::mutex> lock(sProfileMutex);

        sTotalMs += totalMs;
        sUpMs += upMs;
        sModelMs += modelMs;
        sDownMs += downMs;

        if (blocks == 1 || (blocks % 128ULL) == 0ULL)
        {
          const double denom = std::max(1.0, sTotalMs);

          std::ostringstream oss;
          oss << "[NAM Resampler Profile]"
              << " mode=" << modeLabel
              << " blocks=" << blocks
              << " nFrames=" << nFrames
              << " upsampledFrames=" << upsampledFrames
              << " stages=" << mCascadedHalfBandStages
              << " preIIRSections=" << mUpsamplingInputIIRSections.size()
              << " total_ms/block=" << (sTotalMs / static_cast<double>(blocks))
              << " up_ms/block=" << (sUpMs / static_cast<double>(blocks))
              << " model_ms/block=" << (sModelMs / static_cast<double>(blocks))
              << " down_ms/block=" << (sDownMs / static_cast<double>(blocks))
              << " up=" << (100.0 * sUpMs / denom) << "%"
              << " model=" << (100.0 * sModelMs / denom) << "%"
              << " down=" << (100.0 * sDownMs / denom) << "%"
              << "\n";

          std::cerr << oss.str();

          const char* tempDir = std::getenv("TEMP");
          if (tempDir != nullptr && tempDir[0] != '\0')
          {
            std::ofstream f(std::string(tempDir) + "\\NAM-Oversampler-resampler-profile.log", std::ios::app);
            if (f.is_open())
              f << oss.str();
          }

          std::ofstream fLocal("NAM-Oversampler-resampler-profile.log", std::ios::app);
          if (fLocal.is_open())
            fLocal << oss.str();
        }
      }
    }
  }






  void ProcessBlockLinearCascadedFIR(T** inputs, T** outputs, int nFrames, BlockProcessFunc func)
  {
    const bool profile =
      []()
      {
        const char* v = std::getenv("NAM_RESAMPLER_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0' && v[0] != 'f' && v[0] != 'F'
               && v[0] != 'n' && v[0] != 'N';
      }();

    using ProfileClock = std::chrono::high_resolution_clock;
    const auto t0 = profile ? ProfileClock::now() : ProfileClock::time_point{};

    mOutputWritePos = 0;

    const auto tUp0 = profile ? ProfileClock::now() : ProfileClock::time_point{};

    const size_t upsampledFrames =
      CascadedHalfBandUpsampleBlock(PrepareUpsamplerInput(inputs, static_cast<size_t>(nFrames)),
                                      static_cast<size_t>(nFrames),
                                    mEncapsulatedInputPointers.GetList(),
                                    static_cast<size_t>(mMaxEncapsulatedBlockSize));

    const auto tUp1 = profile ? ProfileClock::now() : ProfileClock::time_point{};

    if (upsampledFrames > static_cast<size_t>(mMaxEncapsulatedBlockSize))
      throw std::runtime_error("Cascaded half-band upsampler produced more samples than the model buffer can hold.");

    const auto tModel0 = profile ? ProfileClock::now() : ProfileClock::time_point{};

    func(mEncapsulatedInputPointers.GetList(), mEncapsulatedOutputPointers.GetList(), static_cast<int>(upsampledFrames));

    const auto tModel1 = profile ? ProfileClock::now() : ProfileClock::time_point{};
    const auto tDown0 = profile ? ProfileClock::now() : ProfileClock::time_point{};

    CascadedHalfBandDecimateBlock(mEncapsulatedOutputPointers.GetList(), upsampledFrames, outputs, nFrames);

    const auto tDown1 = profile ? ProfileClock::now() : ProfileClock::time_point{};

    const auto populated = mOutputWritePos;
    if (populated < static_cast<size_t>(nFrames))
    {
      std::cerr << "Cascaded half-band path yielded too few samples (" << populated << " / " << nFrames
                << "). Filling with last sample..." << std::endl;

      for (int c = 0; c < NCHANS; c++)
      {
        const T lastSample = populated > 0 ? outputs[c][populated - 1] : T(0.0);
        for (int i = static_cast<int>(populated); i < nFrames; i++)
          outputs[c][i] = lastSample;
      }
    }

    if (profile)
    {
      const auto t1 = ProfileClock::now();

      static std::atomic<unsigned long long> sBlocks{0};
      static std::atomic<unsigned long long> sInputFrames{0};
      static std::atomic<unsigned long long> sUpsampledFrames{0};
      static std::atomic<unsigned long long> sUpNs{0};
      static std::atomic<unsigned long long> sModelNs{0};
      static std::atomic<unsigned long long> sDownNs{0};
      static std::atomic<unsigned long long> sTotalNs{0};

      const auto nsUp =
        static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(tUp1 - tUp0).count());
      const auto nsModel =
        static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(tModel1 - tModel0).count());
      const auto nsDown =
        static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(tDown1 - tDown0).count());
      const auto nsTotal =
        static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

      sUpNs.fetch_add(nsUp, std::memory_order_relaxed);
      sModelNs.fetch_add(nsModel, std::memory_order_relaxed);
      sDownNs.fetch_add(nsDown, std::memory_order_relaxed);
      sTotalNs.fetch_add(nsTotal, std::memory_order_relaxed);
      sInputFrames.fetch_add(static_cast<unsigned long long>(std::max(0, nFrames)), std::memory_order_relaxed);
      sUpsampledFrames.fetch_add(static_cast<unsigned long long>(upsampledFrames), std::memory_order_relaxed);
      const auto blocks = sBlocks.fetch_add(1, std::memory_order_relaxed) + 1ULL;

      if (blocks == 1ULL || (blocks % 128ULL) == 0ULL)
      {
        const double upMs = static_cast<double>(sUpNs.load(std::memory_order_relaxed)) / 1000000.0;
        const double modelMs = static_cast<double>(sModelNs.load(std::memory_order_relaxed)) / 1000000.0;
        const double downMs = static_cast<double>(sDownNs.load(std::memory_order_relaxed)) / 1000000.0;
        const double totalMs = std::max(0.000001, static_cast<double>(sTotalNs.load(std::memory_order_relaxed)) / 1000000.0);
        const double invBlocks = 1.0 / static_cast<double>(blocks);
        const auto inFrames = sInputFrames.load(std::memory_order_relaxed);
        const auto osFrames = sUpsampledFrames.load(std::memory_order_relaxed);
        const double avgOS = inFrames > 0ULL ? static_cast<double>(osFrames) / static_cast<double>(inFrames) : 0.0;

        std::ostringstream line;
        line << "[NAM Resampler Profile] blocks=" << blocks
             << " avgOS=" << avgOS
             << " total_ms/block=" << (totalMs * invBlocks)
             << " up=" << (100.0 * upMs / totalMs) << "%"
             << " model=" << (100.0 * modelMs / totalMs) << "%"
             << " down=" << (100.0 * downMs / totalMs) << "%"
             << " other=" << (100.0 * std::max(0.0, totalMs - upMs - modelMs - downMs) / totalMs) << "%"
             << std::endl;

        std::cerr << line.str();

        const char* tempDir = std::getenv("TEMP");
        if (tempDir == nullptr || tempDir[0] == '\0')
          tempDir = std::getenv("TMP");

        static std::mutex sNAMProfileLogMutex;
        std::lock_guard<std::mutex> lock(sNAMProfileLogMutex);

        if (tempDir != nullptr && tempDir[0] != '\0')
        {
          const std::string logPath = std::string(tempDir) + "\\NAM-Oversampler-resampler-profile.log";
          std::ofstream logFile(logPath, std::ios::app);
          if (logFile.is_open())
            logFile << line.str();
        }

        // Fallback relative to the current working directory. Useful if TEMP/TMP
        // is unavailable or the host is sandboxing environment variables.
        std::ofstream localLog("NAM-Oversampler-resampler-profile.log", std::ios::app);
        if (localLog.is_open())
          localLog << line.str();
      }
    }
  }



  size_t HalfBandDecimateBy2Stage(int stage, T** input, size_t nFrames, T** output, size_t maxOutputFrames,
                                  bool finalStage)
  {
    // Exact FIR decimation by 2 with prefix/steady-state split.
    //
    // The first historyLen emitted positions may read previous block history.
    // After that, all taps read the current input block and the hot loop is
    // branch-free. Coefficients/cutoff/phase are unchanged.
    const size_t numTaps = mHalfBandCoefficients.size();
    const size_t historyLen = numTaps - 1;
    const size_t stageHistoryOffset = static_cast<size_t>(stage) * NCHANS * historyLen;
    const size_t stageIndex = static_cast<size_t>(stage);

    size_t localOutputWritePos = 0;

    const int startPhase = mCascadedHalfBandPhase[stageIndex] & 1;
    for (size_t s = static_cast<size_t>(startPhase); s < nFrames; s += 2)
    {
      const bool canWrite = output != nullptr
                            && ((!finalStage && localOutputWritePos < maxOutputFrames)
                                || (finalStage && mOutputWritePos < static_cast<size_t>(maxOutputFrames)));

      if (canWrite)
      {
        const size_t writePos = finalStage ? mOutputWritePos : localOutputWritePos;
        const bool steady = s >= historyLen;

        for (int chan = 0; chan < NCHANS; chan++)
        {
          T y = T(0.0);
          T* history = mCascadedHalfBandHistory.data() + stageHistoryOffset + chan * historyLen;
          T* in = input[chan];

          if (steady)
          {
            if constexpr (std::is_same_v<T, float>)
            {
              // SIMD path: reversed coefficients + forward sample window give same dot product.
              y = DenseFIRDotProduct(mHalfBandCoefficientsReversed.data(), &in[s - numTaps + 1], numTaps);
            }
            else
            {
              size_t k = 0;
              for (; k + 3 < numTaps; k += 4)
              {
                y += mHalfBandCoefficients[k + 0] * in[s - (k + 0)];
                y += mHalfBandCoefficients[k + 1] * in[s - (k + 1)];
                y += mHalfBandCoefficients[k + 2] * in[s - (k + 2)];
                y += mHalfBandCoefficients[k + 3] * in[s - (k + 3)];
              }
              for (; k < numTaps; k++)
                y += mHalfBandCoefficients[k] * in[s - k];
            }
          }
          else
          {
            const size_t inputTapCount = std::min(numTaps, s + 1);

            for (size_t k = 0; k < inputTapCount; k++)
              y += mHalfBandCoefficients[k] * in[s - k];

            for (size_t k = inputTapCount; k < numTaps; k++)
            {
              const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
              y += mHalfBandCoefficients[k] * history[historyLen + inputIndex];
            }
          }

          output[chan][writePos] = y;
        }

        if (finalStage)
          mOutputWritePos++;
      }

      if (!finalStage || output == nullptr)
        localOutputWritePos++;
      else
        localOutputWritePos = mOutputWritePos;
    }

    // Equivalent to toggling once per consumed high-rate input sample.
    mCascadedHalfBandPhase[stageIndex] = (mCascadedHalfBandPhase[stageIndex] + static_cast<int>(nFrames & 1U)) & 1;

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = mCascadedHalfBandHistory.data() + stageHistoryOffset + chan * historyLen;

      if (nFrames >= historyLen)
      {
        std::copy(input[chan] + nFrames - historyLen, input[chan] + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input[chan], input[chan] + nFrames, history + historyLen - nFrames);
      }
    }

    return localOutputWritePos;
  }















  void CascadedHalfBandDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    T** stageInput = input;
    size_t stageInputFrames = nFrames;

    std::vector<T*> previousStagePointers(static_cast<size_t>(NCHANS));

    for (int stage = 0; stage < mCascadedHalfBandStages; stage++)
    {
      const bool finalStage = stage == (mCascadedHalfBandStages - 1);
      const size_t maxStageOutputFrames = (stageInputFrames + 1) / 2 + 2;

      if (finalStage)
      {
        HalfBandDecimateBy2Stage(stage, stageInput, stageInputFrames, outputs,
                                 static_cast<size_t>(std::max(0, maxOutputFrames)), true);
      }
      else
      {
        auto& buffer = mCascadedHalfBandBuffers[static_cast<size_t>(stage)];
        const size_t requiredBufferSize = static_cast<size_t>(NCHANS) * maxStageOutputFrames;
        if (buffer.size() < requiredBufferSize)
          buffer.resize(requiredBufferSize);

        std::vector<T*> stageOutputPointers(static_cast<size_t>(NCHANS));
        for (int chan = 0; chan < NCHANS; chan++)
          stageOutputPointers[static_cast<size_t>(chan)] = buffer.data() + chan * maxStageOutputFrames;

        const size_t produced = HalfBandDecimateBy2Stage(stage, stageInput, stageInputFrames,
                                                         stageOutputPointers.data(), maxStageOutputFrames, false);

        previousStagePointers = std::move(stageOutputPointers);
        stageInput = previousStagePointers.data();
        stageInputFrames = produced;
      }
    }
  }

  int GetDecimationFirLatency() const
  {
    if (mDecimationFirCoefficients.empty() || mIntegerDownsampleFactor <= 1)
      return 0;
    return static_cast<int>((mDecimationFirCoefficients.size() - 1) / (2 * mIntegerDownsampleFactor));
  }

  
void DecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    if (UsesCascadedFIR() && !mHalfBandCoefficients.empty())
      CascadedHalfBandDecimateBlock(input, nFrames, outputs, maxOutputFrames);
    else if (!mDecimationFirCoefficients.empty())
    {
      if (mFilterPhase == EAntiAliasFilterPhase::LinearCascadedFIRShort)
        DirectPolyphaseFIRDecimateBlock(input, nFrames, outputs, maxOutputFrames);
      else
        DirectFIRDecimateBlock(input, nFrames, outputs, maxOutputFrames);
    }
    else
    {
      DirectDecimateBlock(input, nFrames, outputs, maxOutputFrames);
    }
  }

  size_t MinimumPhaseIIRTruePolyphaseDecimateBy2Stage(int stage, T** input, size_t nFrames, T** output,
                                                       size_t maxOutputFrames, bool finalStage)
  {
    if (mIIRPolyDecimEvenState.empty() || mIIRPolyDecimOddState.empty())
      EnsureIIRPolyphaseDecimatorState();

    size_t localOutputWritePos = 0;
    const size_t pairs = nFrames / 2;

    for (size_t n = 0; n < pairs; n++)
    {
      if (finalStage && mOutputWritePos >= static_cast<size_t>(maxOutputFrames))
        break;

      if (!finalStage && localOutputWritePos >= maxOutputFrames)
        break;

      const size_t evenIndex = n * 2;
      const size_t oddIndex = evenIndex + 1;
      const size_t writePos = finalStage ? mOutputWritePos : localOutputWritePos;

      for (int chan = 0; chan < NCHANS; chan++)
      {
        const T even = ProcessIIRHalfBandPolyphaseDecimEvenBranch(stage, chan, input[chan][evenIndex]);
        const T odd = ProcessIIRHalfBandPolyphaseDecimOddBranch(stage, chan, input[chan][oddIndex]);

        T y = static_cast<T>(0.5) * (even + odd);

        if (finalStage)
          y = ProcessMinimumPhaseOutputModelStrictIIRGuardSample(chan, y);

        output[chan][writePos] = y;
      }

      if (finalStage)
        mOutputWritePos++;
      else
        localOutputWritePos++;
    }

    return finalStage ? mOutputWritePos : localOutputWritePos;
  }

  void CascadedMinimumPhaseIIRTruePolyphaseDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    if (mCascadedHalfBandStages <= 0)
      return;

    if (mIIRPolyDecimEvenState.empty() || mIIRPolyDecimOddState.empty())
      EnsureIIRPolyphaseDecimatorState();

    if (mIIRHalfBandBuffers.size() < static_cast<size_t>(mCascadedHalfBandStages))
      mIIRHalfBandBuffers.resize(static_cast<size_t>(mCascadedHalfBandStages));

    T** stageInput = input;
    size_t stageInputFrames = nFrames;
    std::array<T*, NCHANS> previousStagePointers {};

    for (int stage = 0; stage < mCascadedHalfBandStages; stage++)
    {
      const bool finalStage = stage == (mCascadedHalfBandStages - 1);
      const size_t maxStageOutputFrames = (stageInputFrames / 2) + 2;

      if (finalStage)
      {
        MinimumPhaseIIRTruePolyphaseDecimateBy2Stage(stage,
                                                     stageInput,
                                                     stageInputFrames,
                                                     outputs,
                                                     static_cast<size_t>(std::max(0, maxOutputFrames)),
                                                     true);
      }
      else
      {
        auto& buffer = mIIRHalfBandBuffers[static_cast<size_t>(stage)];
        const size_t requiredBufferSize = static_cast<size_t>(NCHANS) * maxStageOutputFrames;

        if (buffer.size() < requiredBufferSize)
          buffer.resize(requiredBufferSize);

        std::array<T*, NCHANS> stageOutputPointers {};
        for (int chan = 0; chan < NCHANS; chan++)
          stageOutputPointers[static_cast<size_t>(chan)] =
            buffer.data() + static_cast<size_t>(chan) * maxStageOutputFrames;

        const size_t produced =
          MinimumPhaseIIRTruePolyphaseDecimateBy2Stage(stage,
                                                       stageInput,
                                                       stageInputFrames,
                                                       stageOutputPointers.data(),
                                                       maxStageOutputFrames,
                                                       false);

        previousStagePointers = stageOutputPointers;
        stageInput = previousStagePointers.data();
        stageInputFrames = produced;
      }
    }
  }

  size_t MinimumPhaseIIRFusedDecimateBy2Stage(int stage, T** input, size_t nFrames, T** output,
                                                   size_t maxOutputFrames, bool finalStage)
  {
    // Minimum Phase fused IIR decimator stage.
    //
    // This preserves the verified response: every high-rate sample updates the
    // Butterworth biquad cascade state, but only the selected decimation phase
    // is written to the next stage/output. Do not replace this with the old
    // all-pass half-band decimator: that path had the wrong response/aliasing.
    if (mMinPhaseDownIIRSections.empty())
      DesignMinimumPhaseIIRDownsampleFilter();

    if (mMinPhaseDownIIRSections.empty())
      return 0;

    const size_t numSections = mMinPhaseDownIIRSections.size();
    const size_t stageCount = static_cast<size_t>(std::max(0, mCascadedHalfBandStages));
    const size_t requiredStateCount = stageCount * static_cast<size_t>(NCHANS) * numSections;

    if (mMinPhaseDownIIRState.size() < requiredStateCount)
      mMinPhaseDownIIRState.assign(requiredStateCount, BiquadState {});

    if (mMinPhaseDownIIRPhase.size() < stageCount)
      mMinPhaseDownIIRPhase.assign(stageCount, 0);

    size_t localOutputWritePos = 0;
    int phase = mMinPhaseDownIIRPhase[static_cast<size_t>(stage)] & 1;

    for (size_t s = 0; s < nFrames; s++)
    {
      const bool emit = phase == 0;

      if (emit)
      {
        if (finalStage && mOutputWritePos >= static_cast<size_t>(maxOutputFrames))
          break;

        if (!finalStage && localOutputWritePos >= maxOutputFrames)
          break;
      }

      const size_t writePos = finalStage ? mOutputWritePos : localOutputWritePos;

      for (int chan = 0; chan < NCHANS; chan++)
      {
        T y = input[chan][s];

        const size_t base =
          (static_cast<size_t>(stage) * static_cast<size_t>(NCHANS) + static_cast<size_t>(chan)) * numSections;

        for (size_t section = 0; section < numSections; section++)
        {
          const auto& coeffs = mMinPhaseDownIIRSections[section];
          auto& state = mMinPhaseDownIIRState[base + section];

          const T out = coeffs.b0 * y + state.z1;
          state.z1 = coeffs.b1 * y - coeffs.a1 * out + state.z2;
          state.z2 = coeffs.b2 * y - coeffs.a2 * out;
          y = out;

          if (!std::isfinite(static_cast<double>(y)) || std::abs(static_cast<double>(y)) > 1.0e12)
          {
            for (size_t resetSection = 0; resetSection < numSections; resetSection++)
              mMinPhaseDownIIRState[base + resetSection] = {};
            y = T(0.0);
            break;
          }
        }

        if (emit && output != nullptr)
          output[chan][writePos] = y;
      }

      if (emit)
      {
        if (finalStage)
          mOutputWritePos++;
        else
          localOutputWritePos++;
      }

      phase ^= 1;
    }

    mMinPhaseDownIIRPhase[static_cast<size_t>(stage)] = phase;

    return finalStage ? mOutputWritePos : localOutputWritePos;
  }

  void CascadedMinimumPhaseIIRFusedDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    if (mCascadedHalfBandStages <= 0)
      return;

    if (mMinPhaseDownIIRSections.empty())
      DesignMinimumPhaseIIRDownsampleFilter();

    if (mIIRHalfBandBuffers.size() < static_cast<size_t>(mCascadedHalfBandStages))
      mIIRHalfBandBuffers.resize(static_cast<size_t>(mCascadedHalfBandStages));

    T** stageInput = input;
    size_t stageInputFrames = nFrames;
    std::array<T*, NCHANS> previousStagePointers {};

    for (int stage = 0; stage < mCascadedHalfBandStages; stage++)
    {
      const bool finalStage = stage == (mCascadedHalfBandStages - 1);
      const size_t maxStageOutputFrames = (stageInputFrames + 1) / 2 + 2;

      if (finalStage)
      {
        MinimumPhaseIIRFusedDecimateBy2Stage(stage,
                                             stageInput,
                                             stageInputFrames,
                                             outputs,
                                             static_cast<size_t>(std::max(0, maxOutputFrames)),
                                             true);
      }
      else
      {
        auto& buffer = mIIRHalfBandBuffers[static_cast<size_t>(stage)];
        const size_t requiredBufferSize = static_cast<size_t>(NCHANS) * maxStageOutputFrames;

        if (buffer.size() < requiredBufferSize)
          buffer.resize(requiredBufferSize);

        std::array<T*, NCHANS> stageOutputPointers {};
        for (int chan = 0; chan < NCHANS; chan++)
          stageOutputPointers[static_cast<size_t>(chan)] =
            buffer.data() + static_cast<size_t>(chan) * maxStageOutputFrames;

        const size_t produced =
          MinimumPhaseIIRFusedDecimateBy2Stage(stage,
                                               stageInput,
                                               stageInputFrames,
                                               stageOutputPointers.data(),
                                               maxStageOutputFrames,
                                               false);

        previousStagePointers = stageOutputPointers;
        stageInput = previousStagePointers.data();
        stageInputFrames = produced;
      }
    }
  }





  void DirectFIRDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    const size_t numTaps = mDecimationFirCoefficients.size();
    const size_t historyLen = numTaps - 1;

    for (size_t s = 0; s < nFrames; s++)
    {
      if (mDecimationPhase == 0)
      {
        if (outputs != nullptr && mOutputWritePos < static_cast<size_t>(maxOutputFrames))
        {
          for (int chan = 0; chan < NCHANS; chan++)
          {
            T y = T(0.0);
            T* history = mDecimationFirHistory.data() + chan * historyLen;
            for (size_t k = 0; k < numTaps; k++)
            {
              const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
              const T x = inputIndex >= 0 ? input[chan][inputIndex] : history[historyLen + inputIndex];
              y += mDecimationFirCoefficients[k] * x;
            }
            outputs[chan][mOutputWritePos] = y;
          }
          mOutputWritePos++;
        }
      }
      mDecimationPhase++;
      if (mDecimationPhase >= mIntegerDownsampleFactor)
        mDecimationPhase = 0;
    }

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = mDecimationFirHistory.data() + chan * historyLen;
      if (nFrames >= historyLen)
      {
        std::copy(input[chan] + nFrames - historyLen, input[chan] + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input[chan], input[chan] + nFrames, history + historyLen - nFrames);
      }
    }
  }

  void DirectPolyphaseFIRDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    const int factor = std::max(1, mIntegerDownsampleFactor);
    const size_t numTaps = mDecimationFirCoefficients.size();
    const size_t historyLen = numTaps - 1;

    for (size_t s = 0; s < nFrames; s++)
    {
      if (mDecimationPhase == 0)
      {
        if (outputs != nullptr && mOutputWritePos < static_cast<size_t>(maxOutputFrames))
        {
          for (int chan = 0; chan < NCHANS; chan++)
          {
            T y = T(0.0);
            T* history = mDecimationFirHistory.data() + chan * historyLen;
            for (int phase = 0; phase < factor; phase++)
            {
              for (size_t k = static_cast<size_t>(phase); k < numTaps; k += static_cast<size_t>(factor))
              {
                const long inputIndex = static_cast<long>(s) - static_cast<long>(k);
                const T x = inputIndex >= 0 ? input[chan][inputIndex] : history[historyLen + inputIndex];
                y += mDecimationFirCoefficients[k] * x;
              }
            }
            outputs[chan][mOutputWritePos] = y;
          }
          mOutputWritePos++;
        }
      }
      mDecimationPhase++;
      if (mDecimationPhase >= mIntegerDownsampleFactor)
        mDecimationPhase = 0;
    }

    for (int chan = 0; chan < NCHANS; chan++)
    {
      T* history = mDecimationFirHistory.data() + chan * historyLen;
      if (nFrames >= historyLen)
      {
        std::copy(input[chan] + nFrames - historyLen, input[chan] + nFrames, history);
      }
      else
      {
        std::move(history + nFrames, history + historyLen, history);
        std::copy(input[chan], input[chan] + nFrames, history + historyLen - nFrames);
      }
    }
  }

  void DirectDecimateBlock(T** input, size_t nFrames, T** outputs, int maxOutputFrames)
  {
    for (size_t s = 0; s < nFrames; s++)
    {
      if (mDecimationPhase == 0)
      {
        if (outputs != nullptr && mOutputWritePos < static_cast<size_t>(maxOutputFrames))
        {
          for (int chan = 0; chan < NCHANS; chan++)
            outputs[chan][mOutputWritePos] = input[chan][s];
          mOutputWritePos++;
        }
      }
      mDecimationPhase++;
      if (mDecimationPhase >= mIntegerDownsampleFactor)
        mDecimationPhase = 0;
    }
  }

  // Buffers for scratch input data for Reset() to use
  WDL_TypedBuf<T> mScratchExternalInputData;
  WDL_PtrList<T> mScratchExternalInputPointers;
  // Buffers for the input & output to the encapsulated DSP
  WDL_TypedBuf<T> mEncapsulatedInputData;
  WDL_PtrList<T> mEncapsulatedInputPointers;
  WDL_TypedBuf<T> mEncapsulatedOutputData;
  WDL_PtrList<T> mEncapsulatedOutputPointers;
  WDL_TypedBuf<T> mAntiAliasOutputData;
  WDL_PtrList<T> mAntiAliasOutputPointers;
  std::vector<T> mAntiAliasCoefficients;
  std::vector<T> mAntiAliasHistory;
  std::vector<T> mUpsamplingInputFilterCoefficients;
  std::vector<T> mUpsamplingInputFilterHistory;
  std::vector<BiquadCoefficients> mUpsamplingInputIIRSections;
  std::vector<BiquadState> mUpsamplingInputIIRState;
  std::vector<BiquadCoefficients> mMinimumPhaseSections;
  std::vector<BiquadState> mMinimumPhaseState;
  std::vector<T> mDecimationFirCoefficients;
  std::vector<T> mDecimationFirHistory;
  std::vector<T> mHalfBandCoefficients;
  std::vector<T> mHalfBandCoefficientsReversed; // mHalfBandCoefficients reversed, for SIMD forward-access
  std::array<std::vector<size_t>, 2> mHalfBandPolyphaseTapIndices;
  std::array<std::vector<T>, 2> mHalfBandPolyphaseTapCoefficients;
  std::vector<T> mCascadedHalfBandHistory;
  std::vector<int> mCascadedHalfBandPhase;
  std::vector<std::vector<T>> mCascadedHalfBandBuffers;
  std::vector<T> mCascadedHalfBandInterpHistory;
  std::vector<std::vector<T>> mCascadedHalfBandInterpBuffers;
  std::vector<std::vector<T>> mIIRHalfBandBuffers;
  std::vector<OnePoleAllpassState> mIIRInterpAState;
  std::vector<OnePoleAllpassState> mIIRInterpBState;
  std::vector<OnePoleAllpassState> mIIRPolyDecimEvenState;
  std::vector<OnePoleAllpassState> mIIRPolyDecimOddState;
  int mCascadedHalfBandStages = 0;
  bool mAntiAliasEnabled = false;
  EAntiAliasFilterPhase mFilterPhase = EAntiAliasFilterPhase::MinimumPhaseCascadedFIR;
  EAntiAliasFilterPhase mDesignedFilterPhase = EAntiAliasFilterPhase::MinimumPhaseCascadedFIR;
  bool mUseIntegerDownsampler = false;
  bool mUseCascadedHalfBandResampler = false;
  bool mUseRealtimeIIRHalfBand = false;
  int mIntegerDownsampleFactor = 1;
  int mDecimationPhase = 0;
  size_t mOutputWritePos = 0;
  // Sample rate ratio from external to encapsulated, from encapsulated to external.
  double mRatio1 = 0.0, mRatio2 = 0.0;
  // Sample rate of the external context.
  std::vector<BiquadCoefficients> mMinPhaseDownIIRSections;
  std::vector<BiquadCoefficients> mMinPhaseOutputStrictIIRSections;
  std::vector<BiquadState> mMinPhaseOutputStrictIIRState;
  std::vector<BiquadState> mMinPhaseDownIIRState;
  std::vector<int> mMinPhaseDownIIRPhase;
  double mInputSampleRate = 0.0;
  // The size of the largest block the external context may provide. (It might provide something smaller.)
  int mMaxBlockSize = 0;
  // The size of the largest possible encapsulated block
  int mMaxEncapsulatedBlockSize = 0;
  // How much latency this object adds due to both of its resamplers. This does _not_ include the latency due to the
  // encapsulated `func()`.
  int mLatency = 0;
  // The sample rate required by the DSP that this object encapsulates
  const double mRenderingSampleRate;
  // The highest sample rate whose bandwidth should be preserved by reconstruction filters.
  const double mBandwidthSampleRate;
  // Pair of resamplers for (1) external -> encapsulated, (2) encapsulated -> external
  std::unique_ptr<LanczosResamplerType> mResampler1, mResampler2;
};

}; // namespace dsp
