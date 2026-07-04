#pragma once

#include <cstddef>

namespace cosq::qnet {
namespace blas {

void sgemm(bool transA, bool transB,
           size_t M, size_t N, size_t K,
           float alpha,
           const float* A, size_t lda,
           const float* B, size_t ldb,
           float beta,
           float* C, size_t ldc);

void sgemm_batch(bool transA, bool transB,
                 size_t M, size_t N, size_t K,
                 float alpha,
                 const float* const* A, size_t lda,
                 const float* const* B, size_t ldb,
                 float beta,
                 float** C, size_t ldc,
                 size_t batch_count);

} // namespace blas
} // namespace cosq::qnet
