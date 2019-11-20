/*
 * @Author: Haozhe Xie
 * @Date:   2019-11-13 10:53:22
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2019-11-20 11:47:16
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
                                const float *__restrict__ ptcloud,
                                float *__restrict__ weights) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  ptcloud += batch_index * n_pts * 3;
  weights += batch_index * n_grid_vertices * n_pts * 3;

  for (int j = index; j < n_pts; j += stride) {
    float pt_x = ptcloud[j * 3 + 0];
    float pt_y = ptcloud[j * 3 + 1];
    float pt_z = ptcloud[j * 3 + 2];

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
    int lx_offset = lower_x - min_x, ux_offset = upper_x - min_x;
    int ly_offset = lower_y - min_y, uy_offset = upper_y - min_y;
    int lz_offset = lower_z - min_z, uz_offset = upper_z - min_z;

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
    n_grid_vertices, n_pts, min_x, min_y, min_z, len_y, len_z,
    ptcloud.data<float>(), grid_weights.data<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return grid_weights;
}

__global__ void gridding_grad_kernel(int n_grid_vertices,
                                     int n_pts,
                                     float min_x,
                                     float min_y,
                                     float min_z,
                                     int len_y,
                                     int len_z,
                                     const float *__restrict__ grad_grid,
                                     const float *__restrict__ ptcloud,
                                     const float *__restrict__ weights,
                                     float *__restrict__ grad_ptcloud) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  grad_grid += batch_index * n_grid_vertices;
  ptcloud += batch_index * n_pts * 3;
  weights += batch_index * n_grid_vertices * n_pts * 3;
  grad_ptcloud += batch_index * n_pts * 3;

  for (int j = index; j < n_pts; j += stride) {
    float pt_x = ptcloud[j * 3 + 0];
    float pt_y = ptcloud[j * 3 + 1];
    float pt_z = ptcloud[j * 3 + 2];

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
    int lx_offset = lower_x - min_x, ux_offset = upper_x - min_x;
    int ly_offset = lower_y - min_y, uy_offset = upper_y - min_y;
    int lz_offset = lower_z - min_z, uz_offset = upper_z - min_z;
    float grad_vtx = 0, x_weights = 0, y_weights = 0, z_weights = 0;

    // Compute gradient for the corresponding positions, a loop for 8 points
    // LLL -> Lower X, Lower Y, Lower Z
    gvtx_idx  = compute_index(lx_offset, ly_offset, lz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // LLU -> Lower X, Lower Y, Upper Z
    gvtx_idx  = compute_index(lx_offset, ly_offset, uz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);

    // LUL -> Lower X, Upper Y, Lower Z
    gvtx_idx  = compute_index(lx_offset, uy_offset, lz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // LUU -> Lower X, Upper Y, Upper Z
    gvtx_idx  = compute_index(lx_offset, uy_offset, uz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), -grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);

    // ULL -> Upper X, Lower Y, Lower Z
    gvtx_idx  = compute_index(ux_offset, ly_offset, lz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // ULU -> Upper X, Lower Y, Upper Z
    gvtx_idx  = compute_index(ux_offset, ly_offset, uz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), -grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);

    // UUL -> Upper X, Upper Y, Lower Z
    gvtx_idx  = compute_index(ux_offset, uy_offset, lz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), -grad_vtx * x_weights * y_weights);

    // UUU -> Upper X, Upper Y, Upper Z
    gvtx_idx  = compute_index(ux_offset, uy_offset, uz_offset, len_y, len_z);
    grad_vtx  = grad_grid[gvtx_idx];
    x_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 0];
    y_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 1];
    z_weights = weights[gvtx_idx * n_pts * 3 + j * 3 + 2];
    atomicAdd(&(grad_ptcloud[j * 3 + 0]), grad_vtx * y_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 1]), grad_vtx * x_weights * z_weights);
    atomicAdd(&(grad_ptcloud[j * 3 + 2]), grad_vtx * x_weights * y_weights);
  }
}

torch::Tensor gridding_cuda_backward(float min_x,
                                     float max_x,
                                     float min_y,
                                     float max_y,
                                     float min_z,
                                     float max_z,
                                     torch::Tensor grad_grid,
                                     torch::Tensor ptcloud,
                                     torch::Tensor grid_weights,
                                     cudaStream_t stream) {
  int batch_size      = grid_weights.size(0);
  int n_grid_vertices = grid_weights.size(1);
  int n_pts           = grid_weights.size(2);
  int len_y           = max_y - min_y + 1;
  int len_z           = max_z - min_z + 1;

  torch::Tensor grad_ptcloud =
    torch::zeros({batch_size, n_pts, 3}, torch::CUDA(torch::kFloat));

  gridding_grad_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    n_grid_vertices, n_pts, min_x, min_y, min_z, len_y, len_z,
    grad_grid.data<float>(), ptcloud.data<float>(), grid_weights.data<float>(),
    grad_ptcloud.data<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_cuda_backward: %s\n", cudaGetErrorString(err));
  }
  return grad_ptcloud;
}