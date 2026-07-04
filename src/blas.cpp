#include <qnet/blas.hpp>

#include <cblas.h>
#include <cstddef>

namespace cosq::qnet {
namespace blas {

void sgemm(bool transA, bool transB,
           size_t M, size_t N, size_t K,
           float alpha,
           const float* A, size_t lda,
           const float* B, size_t ldb,
           float beta,
           float* C, size_t ldc) {
    cblas_sgemm(
        CblasRowMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        static_cast<blasint>(M),
        static_cast<blasint>(N),
        static_cast<blasint>(K),
        alpha,
        A, static_cast<blasint>(lda),
        B, static_cast<blasint>(ldb),
        beta,
        C, static_cast<blasint>(ldc)
    );
}

void sgemm_batch(bool transA, bool transB,
                 size_t M, size_t N, size_t K,
                 float alpha,
                 const float* const* A, size_t lda,
                 const float* const* B, size_t ldb,
                 float beta,
                 float** C, size_t ldc,
                 size_t batch_count) {
    for (size_t i = 0; i < batch_count; ++i) {
        sgemm(transA, transB, M, N, K,
              alpha, A[i], lda, B[i], ldb, beta, C[i], ldc);
    }
}

} // namespace blas
} // namespace cosq::qnet
