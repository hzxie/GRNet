// Copyright 2019 Haozhe Xie and Max Planck Society
// Distributed under the MIT Software license,
// (See https://opensource.org/licenses/MIT)

#include "math_utils.hpp"

inline const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition)                               \
  /* Code block avoids redefinition of cudaError_t error */ \
  do {                                                      \
    cudaError_t error = condition;                          \
    if (error != cudaSuccess) cudaGetErrorString(error);    \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUBLAS_CHECK(condition)                                        \
  do {                                                                 \
    cublasStatus_t status = condition;                                 \
    if (status != CUBLAS_STATUS_SUCCESS) cublasGetErrorString(status); \
  } while (0)

void gpu_scal(const cublasHandle_t& handle,
              const int N,
              const float alpha,
              float* X) {
  CUBLAS_CHECK(cublasSscal(handle, N, &alpha, X, 1));
}

__global__ void mul_kernel(const int n,
                           const float* a,
                           const float* b,
                           float* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] * b[index]; }
}

void gpu_mul(const int N, const float* a, const float* b, float* y) {
  mul_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, b, y);
}

__global__ void mul_inverse_kernel(const int n,
                                   const float* x,
                                   float* y,
                                   float eps) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = 1.0 / (x[index] + eps); }
}

void gpu_mul_inverse(const int N, const float* x, float* y, float eps) {
  mul_inverse_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, x, y, eps);
}

void gpu_gemm(const cublasHandle_t& handle,
              const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB,
              const int M,
              const int N,
              const int K,
              const float alpha,
              const float* A,
              const float* B,
              const float beta,
              float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;

  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb,
                           A, lda, &beta, C, N));
}

void gpu_gemm_ex(const cublasHandle_t& handle,
                 const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB,
                 const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const int lda,
                 const float* B,
                 const int ldb,
                 const float beta,
                 float* C,
                 const int ldc) {
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb,
                           A, lda, &beta, C, ldc));
}
