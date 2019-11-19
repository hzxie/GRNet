/*
 * @Author: Haozhe Xie
 * @Date:   2019-11-13 10:53:22
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2019-11-19 21:27:38
 * @Email:  cshzxie@gmail.com
 */

#include <torch/extension.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<double>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__device__ int compute_index(
  int offset_x, int offset_y, int offset_z, int len_y, int len_z) {
  return offset_x * len_y * len_z + offset_y * len_z + offset_z;
}

__device__ float compute_weight(float x, float x0) { return (1 - abs(x - x0)); }

__global__ void gridding_kernel(int n_grid_vertices,
                                int n_pts,
                                float min_x,
                                float min_y,
                                float min_z,
                                int len_y,
                                int len_z,
                                const float *__restrict__ ptcloud_xyz,
                                float *__restrict__ weights) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  ptcloud_xyz += batch_index * n_pts * 3;
  weights += 3 * n_grid_vertices * n_pts * batch_index;

  for (int j = index; j < n_pts; j += stride) {
    float pt_x = ptcloud_xyz[j * 3 + 0];
    float pt_y = ptcloud_xyz[j * 3 + 1];
    float pt_z = ptcloud_xyz[j * 3 + 2];

    int lower_x = std::floor(pt_x);
    int upper_x = std::ceil(pt_x);
    if (lower_x == upper_x) {
      upper_x += 1;
    }
    int lower_y = std::floor(pt_y);
    int upper_y = std::ceil(pt_y);
    if (lower_y == upper_y) {
      upper_y += 1;
    }
    int lower_z = std::floor(pt_z);
    int upper_z = std::ceil(pt_z);
    if (lower_z == upper_z) {
      upper_z += 1;
    }

    int gvtx_idx  = 0;
    int lx_offset = lower_x - min_x;
    int ly_offset = lower_y - min_y;
    int lz_offset = lower_z - min_z;
    int ux_offset = upper_x - min_x;
    int uy_offset = upper_y - min_y;
    int uz_offset = upper_z - min_z;

    // Compute weights and corresponding positions, a loop for 8 points
    // LLL -> Lower X, Lower Y, Lower Z
    gvtx_idx = compute_index(lx_offset, ly_offset, lz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, lower_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, lower_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, lower_z);

    // LLU -> Lower X, Lower Y, Upper Z
    gvtx_idx = compute_index(lx_offset, ly_offset, uz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, lower_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, lower_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, upper_z);

    // LUL -> Lower X, Upper Y, Lower Z
    gvtx_idx = compute_index(lx_offset, uy_offset, lz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, lower_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, upper_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, lower_z);

    // LUU -> Lower X, Upper Y, Upper Z
    gvtx_idx = compute_index(lx_offset, uy_offset, uz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, lower_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, upper_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, upper_z);

    // ULL -> Upper X, Lower Y, Lower Z
    gvtx_idx = compute_index(ux_offset, ly_offset, lz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, upper_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, lower_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, lower_z);

    // ULU -> Upper X, Lower Y, Upper Z
    gvtx_idx = compute_index(ux_offset, ly_offset, uz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, upper_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, lower_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, upper_z);

    // UUL -> Upper X, Upper Y, Lower Z
    gvtx_idx = compute_index(ux_offset, uy_offset, lz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, upper_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, upper_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, lower_z);

    // UUU -> Upper X, Upper Y, Upper Z
    gvtx_idx = compute_index(ux_offset, uy_offset, uz_offset, len_y, len_z);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 0] = compute_weight(pt_x, upper_x);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 1] = compute_weight(pt_y, upper_y);
    weights[gvtx_idx * n_pts * 3 + j * 3 + 2] = compute_weight(pt_z, upper_z);
  }
}

torch::Tensor gridding_cuda_forward(float min_x,
                                    float max_x,
                                    float min_y,
                                    float max_y,
                                    float min_z,
                                    float max_z,
                                    torch::Tensor ptcloud,
                                    cudaStream_t stream) {
  int batch_size      = ptcloud.size(0);
  int n_pts           = ptcloud.size(1);
  int len_x           = max_x - min_x + 1;
  int len_y           = max_y - min_y + 1;
  int len_z           = max_z - min_z + 1;
  int n_grid_vertices = len_x * len_y * len_z;

  torch::Tensor grid_weights = torch::zeros(
    {batch_size, n_grid_vertices, n_pts, 3}, torch::CUDA(torch::kFloat));

  gridding_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    batch_size, n_pts, min_x, min_y, min_z, len_y, len_z, ptcloud.data<float>(),
    grid_weights.data<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return grid_weights;
}