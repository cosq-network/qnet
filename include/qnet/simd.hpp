#pragma once

#include <cstddef>
#include <cmath>

#if defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#include <immintrin.h>
#endif

namespace cosq::qnet {
namespace simd {

inline void relu(float* dst, const float* src, size_t n) {
    size_t i = 0;

#if defined(__ARM_NEON__)
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        float32x4_t r = vmaxq_f32(x, zero);
        vst1q_f32(dst + i, r);
    }
#elif defined(__AVX__)
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 r = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(dst + i, r);
    }
#elif defined(__SSE2__)
    __m128 zero = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 x = _mm_loadu_ps(src + i);
        __m128 r = _mm_max_ps(x, zero);
        _mm_storeu_ps(dst + i, r);
    }
#endif

    for (; i < n; ++i) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}

inline void add(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;

#if defined(__ARM_NEON__)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vaddq_f32(va, vb));
    }
#elif defined(__AVX__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(va, vb));
    }
#elif defined(__SSE2__)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(dst + i, _mm_add_ps(va, vb));
    }
#endif

    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
}

inline void mul(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;

#if defined(__ARM_NEON__)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(dst + i, vmulq_f32(va, vb));
    }
#elif defined(__AVX__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(va, vb));
    }
#elif defined(__SSE2__)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(dst + i, _mm_mul_ps(va, vb));
    }
#endif

    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
}

inline void sigmoid(float* dst, const float* src, size_t n) {
    size_t i = 0;

#if defined(__ARM_NEON__)
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        float32x4_t neg_x = vnegq_f32(x);

        float32x4_t exp_neg;
        int32x4_t yi = vcvtq_s32_f32(vmulq_n_f32(neg_x, 1.4426950408889634f));
        float32x4_t yf = vsubq_f32(
            vmulq_n_f32(neg_x, 1.4426950408889634f),
            vcvtq_f32_s32(yi));

        float32x4_t p = vdupq_n_f32(0.051968f);
        p = vfmaq_f32(vdupq_n_f32(0.241379f), p, yf);
        p = vfmaq_f32(vdupq_n_f32(0.693032f), p, yf);
        p = vfmaq_f32(one, p, yf);

        int32x4_t shift = vshlq_s32(yi, vdupq_n_s32(23));
        float32x4_t exp2 = vreinterpretq_f32_s32(
            vaddq_s32(shift, vdupq_n_s32(127 << 23)));
        exp_neg = vmulq_f32(exp2, p);

        float32x4_t denom = vaddq_f32(one, exp_neg);
        float32x4_t r = vrecpeq_f32(denom);

        r = vmulq_f32(vrecpsq_f32(denom, r), r);
        r = vmulq_f32(vrecpsq_f32(denom, r), r);

        vst1q_f32(dst + i, r);
    }
#elif defined(__AVX__)
    // AVX sigmoid using similar approach or falling back to scalar
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

        __m256 exp_neg;
        __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        __m256 y = _mm256_mul_ps(neg_x, log2e);
        __m256 yi = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 yf = _mm256_sub_ps(y, yi);

        __m256 p = _mm256_set1_ps(0.051968f);
        p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(0.241379f));
        p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(0.693032f));
        p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(1.0f));

        __m256i yi_int = _mm256_cvtps_epi32(yi);
        __m256i shift = _mm256_sllv_epi32(yi_int, _mm256_set1_epi32(23));
        __m256i bias = _mm256_set1_epi32(127 << 23);
        __m256 exp2 = _mm256_castsi256_ps(_mm256_add_epi32(shift, bias));

        exp_neg = _mm256_mul_ps(exp2, p);

        __m256 one = _mm256_set1_ps(1.0f);
        __m256 denom = _mm256_add_ps(one, exp_neg);
        __m256 r = _mm256_div_ps(one, denom);

        _mm256_storeu_ps(dst + i, r);
    }
#endif

    for (; i < n; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

} // namespace simd
} // namespace cosq::qnet
