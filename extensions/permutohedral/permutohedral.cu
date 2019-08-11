// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include <ATen/ATen.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "math_utils.hpp"
#include "permutohedral_cuda.hpp"

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

inline void gpu_memset(const size_t N, const int alpha, void* X) {
  CUDA_CHECK(cudaMemset(X, alpha, N));
}

inline void gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
  }
}

/************************************************/
/***              CUDA Functions              ***/
/************************************************/

__global__ void copy_to_blurred_out_data(const float* sliced_out,
                                         const int M,
                                         const int value_size,
                                         float* blurred_out) {
  CUDA_KERNEL_LOOP(k, value_size) {
    for (int t = 0; t < M; t++) {
      blurred_out[k * (M + 1) + t + 1] += sliced_out[k * M + t];
    }
  }
}

__global__ void copy_to_blur_data(const float* splat_data,
                                  const int M,
                                  const int num_output,
                                  float* blur_data) {
  CUDA_KERNEL_LOOP(k, num_output) {
    for (int t = 0; t < M; t++) {
      blur_data[k * M + t] = splat_data[k * (M + 1) + t + 1];
    }
  }
}

__global__ void slice_gpu_kernel(const int out_size,
                                 const float* data,
                                 const int M,
                                 const int d,
                                 const int out_offset,
                                 const int num_output,
                                 const int* offset,
                                 const float* barycentric,
                                 float* sliced) {
  CUDA_KERNEL_LOOP(i, out_size) {
    for (int k = 0; k < num_output; k++) sliced[k * out_size + i] = 0;

    for (int j = 0; j <= d; j++) {
      int o   = offset[(out_offset + i) * (d + 1) + j];
      float w = barycentric[(out_offset + i) * (d + 1) + j];

      for (int k = 0; k < num_output; k++) {
        sliced[k * out_size + i] += w * data[k * M + o];
      }
    }
  }
}

__global__ void maxpool_gpu_kernel(const float* splat_data,
                                   const int value_size,
                                   const int filter_size,
                                   const int M,
                                   const int* blur_neighbors,
                                   float* maxxed_data,
                                   int* idx_data) {
  CUDA_KERNEL_LOOP(i, value_size) {
    for (int j = 0; j < M; ++j) {
      int idx         = j;
      float max_value = splat_data[i * (M + 1) + idx + 1];
      for (int k = 1; k < filter_size; ++k) {
        const int* neighbors = blur_neighbors + ((k - 1) * M);
        float value          = splat_data[i * (M + 1) + neighbors[j] + 1];
        if (value > max_value) {
          max_value = value;
          idx       = neighbors[j];
        }
      }
      maxxed_data[i * M + j] = max_value;
      idx_data[i * M + j]    = i * (M + 1) + idx + 1;
    }
  }
}

__global__ void splat_gpu_kernel(const int in_size,
                                 const float* in,
                                 const int M,
                                 const int d,
                                 const int in_offset,
                                 const int value_size,
                                 const int* offset,
                                 const float* barycentric,
                                 float* splatted) {
  CUDA_KERNEL_LOOP(i, in_size) {
    for (int j = 0; j <= d; ++j) {
      int o          = offset[(in_offset + i) * (d + 1) + j] + 1;
      const float& w = barycentric[(in_offset + i) * (d + 1) + j];

      for (int k = 0; k < value_size; k++) {
        atomicAdd(&splatted[k * (M + 1) + o], w * in[k * in_size + i]);
      }
    }
  }
}

__global__ void im2col_gpu_kernel(const float* im,
                                  const int value_size,
                                  const int filter_size,
                                  const int M,
                                  const int start,
                                  const int output_size,
                                  const int* blur_neighbors,
                                  float* col) {
  CUDA_KERNEL_LOOP(i, output_size) {
    for (int k = 0; k < value_size; ++k) {
      col[(k * filter_size + 0) * output_size + i] =
        im[k * (M + 1) + (i + start + 1)];

      for (int f = 1; f < filter_size; ++f) {
        const int* neighbors = &blur_neighbors[(f - 1) * M + 0];

        col[(k * filter_size + f) * output_size + i] =
          im[k * (M + 1) + (neighbors[i + start] + 1)];
      }
    }
  }
}

__global__ void col2im_gpu_kernel(const float* col,
                                  const int value_size,
                                  const int filter_size,
                                  const int M,
                                  const int start,
                                  const int output_size,
                                  const int* blur_neighbors,
                                  float* im) {
  CUDA_KERNEL_LOOP(i, output_size) {
    for (int k = 0; k < value_size; ++k) {
      im[k * (M + 1) + (i + start + 1)] +=
        col[(k * filter_size + 0) * output_size + i];

      for (int f = 1; f < filter_size; ++f) {
        const int* neighbors = &blur_neighbors[(f - 1) * M + 0];

        im[k * (M + 1) + (neighbors[i + start] + 1)] +=
          col[(k * filter_size + f) * output_size + i];
      }
    }
  }
}

__global__ void slice_tick_gpu_kernel(const int out_size,
                                      const float* sliced,
                                      const int M,
                                      const int d,
                                      const int out_offset,
                                      const int num_output,
                                      const int* offset,
                                      const float* barycentric,
                                      float* data) {
  CUDA_KERNEL_LOOP(i, out_size) {
    for (int j = 0; j <= d; j++) {
      int o   = offset[(out_offset + i) * (d + 1) + j];
      float w = barycentric[(out_offset + i) * (d + 1) + j];

      for (int k = 0; k < num_output; k++) {
        atomicAdd(&data[k * M + o], w * sliced[k * out_size + i]);
      }
    }
  }
}

__global__ void maxtick_gpu_kernel(const float* tick_data,
                                   const int value_size,
                                   const int filter_size,
                                   const int M,
                                   const int* blur_neighbors,
                                   float* max_out_tick_data,
                                   const int* idx_data) {
  CUDA_KERNEL_LOOP(i, value_size) {
    // Looping over variables
    for (int j = 0; j < M; ++j) {
      // Looping only over the neighbors
      if (idx_data[i * M + j] == i * (M + 1) + j + 1) {
        max_out_tick_data[i * (M + 1) + j + 1] += tick_data[i * M + j];
      }
      for (int k = 1; k < filter_size; ++k) {
        const int* neighbors = blur_neighbors + ((k - 1) * M);
        if (idx_data[i * M + j] == i * (M + 1) + neighbors[j] + 1) {
          max_out_tick_data[i * (M + 1) + neighbors[j] + 1] +=
            tick_data[i * M + j];
        }
      }
    }
  }
}

__global__ void splat_tick_gpu_kernel(const int in_size,
                                      const float* splatted,
                                      const int M,
                                      const int d,
                                      const int in_offset,
                                      const int value_size,
                                      const int* offset,
                                      const float* barycentric,
                                      float* in) {
  CUDA_KERNEL_LOOP(i, in_size) {
    for (int j = 0; j <= d; ++j) {
      int o          = offset[(in_offset + i) * (d + 1) + j] + 1;
      const float& w = barycentric[(in_offset + i) * (d + 1) + j];

      for (int k = 0; k < value_size; k++) {
        in[k * in_size + i] += w * splatted[k * (M + 1) + o];
      }
    }
  }
}

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

void Permutohedral::map_back(const std::vector<boost::int16_t>& key,
                             float* const x) {
  const int d_      = key.size() - 1;
  float inv_std_dev = std::sqrt(2.0 / 3.0) * (d_ + 1);

  std::vector<float> scale_factor(d_);
  for (int i = 0; i < d_; ++i) {
    scale_factor[i] = 1.0 / std::sqrt((i + 2) * (i + 1)) * inv_std_dev;
  }

  float sum = 0;
  for (int j = d_; j > 0; --j) {
    float cf = (sum - key[j]) / j;
    x[j - 1] = cf / scale_factor[j - 1];
    sum += cf;
  }
  assert(std::abs(sum - key[0]) < 1e-3);
}

Permutohedral::Permutohedral() {}

int Permutohedral::get_filter_size(int neighborhood_size, int feature_size) {
  return ::get_filter_size(neighborhood_size, feature_size);
}

void Permutohedral::init(const float* feature,
                         int data_count,
                         int feature_size,
                         int neighborhood_size,
                         bool do_visualization) {
  const int N = data_count;
  const int d = feature_size;

  boost::shared_ptr<Lattice> lattice = boost::make_shared<Lattice>();
  // Set the read only shared lattice data
  lattice_ = lattice;

  lattice->N_                 = N;
  lattice->d_                 = d;
  lattice->neighborhood_size_ = neighborhood_size;

  // Allocate enough storage
  lattice->barycentric_.resize(static_cast<int>((d + 1) * N));
  std::vector<boost::int16_t> ranks;
  ranks.resize((d + 1) * N);
  lattice->offset_.resize((d + 1) * N);

  // Compute the lattice coordinates for each feature [there is going to be
  // a lot of magic here
  HashTable hash_table(d, N * (d + 1));

  // Allocate the local memory
  std::vector<float> scale_factor(d);
  std::vector<float> elevated(d + 1);
  std::vector<float> rem0(d + 1);
  std::vector<float> barycentric(d + 2);
  std::vector<boost::int16_t> canonical((d + 1) * (d + 1));
  std::vector<boost::int16_t> key(d + 1);

  // Compute the canonical simplex
  for (int i = 0; i <= d; i++) {
    for (int j = 0; j <= d - i; j++) canonical[i * (d + 1) + j] = i;
    for (int j = d - i + 1; j <= d; j++)
      canonical[i * (d + 1) + j] = i - (d + 1);
  }

  // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
  float inv_std_dev = sqrt(2.0 / 3.0) * (d + 1);
  // Compute the diagonal part of E (p.5 in [Adams etal 2010])
  for (int i = 0; i < d; i++)
    scale_factor[i] = 1.0 / sqrt((i + 2) * (i + 1)) * inv_std_dev;

  const float* f = feature;

  std::vector<boost::int16_t> min_key(d + 1);
  std::vector<boost::int16_t> max_key(d + 1);

  // Compute the simplex each feature lies in
  for (int k = 0; k < N; k++) {
    // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
    // const float * f = feature + k*feature_size;

    // sm contains the sum of 1..n of our faeture vector
    float sm(0);
    for (int j = d; j > 0; j--) {
      const int fIndex = (j - 1) * N + k;
      float cf         = f[fIndex] * scale_factor[j - 1];
      elevated[j]      = sm - j * cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    float down_factor = 1.0f / (d + 1);
    float up_factor   = (d + 1);
    int sum           = 0;
    for (int i = 0; i <= d; i++) {
      int rd  = round(down_factor * static_cast<float>(elevated[i]));
      rem0[i] = rd * up_factor;
      sum += rd;
    }

    // Find the simplex we are in and store it in rank (where rank
    // describes what position coorinate i has in the sorted order of the
    // features values)
    boost::int16_t* rank = ranks.data() + (d + 1) * k;
    for (int i = 0; i < d; i++) {
      double di = static_cast<float>(elevated[i]) - rem0[i];
      for (int j = i + 1; j <= d; j++)
        if (di < static_cast<float>(elevated[j]) - rem0[j])
          rank[i]++;
        else
          rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= d; i++) {
      rank[i] += sum;
      if (rank[i] < 0) {
        rank[i] += d + 1;
        rem0[i] += d + 1;
      } else if (rank[i] > d) {
        rank[i] -= d + 1;
        rem0[i] -= d + 1;
      }
    }

    // If do_visualization is true, fill barycentric weights with 1.0
    // Otherwise, comptue the barycentric coordinates (p.10 in [Adams et al.
    // 2010])
    if (do_visualization) {
      for (int i = 0; i <= d + 1; i++) {
        barycentric[i] = 1.0;
      }
    } else {
      for (int i = 0; i <= d + 1; i++) barycentric[i] = 0;
      for (int i = 0; i <= d; i++) {
        float v = (elevated[i] - rem0[i]) * down_factor;

        if (d - rank[i] < 0 || d - rank[i] + 1 >= d + 2)
          throw std::runtime_error("Permutohedral: rank access error");

        // assert(d_-rank[i]   >= 0);
        // assert(d_-rank[i]+1 <  d_+2);
        barycentric[d - rank[i]] += v;
        barycentric[d - rank[i] + 1] -= v;
      }
      // Wrap around
      barycentric[0] += 1.0 + barycentric[d + 1];
    }

    // Compute all vertices and their offset
    std::vector<boost::int16_t> neighborKeyUp(d + 1);
    std::vector<boost::int16_t> neighborKeyDown(d + 1);
    for (int remainder = 0; remainder <= d; remainder++) {
      for (int i = 0; i < d; i++)
        key[i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]];
      assert(k * (d + 1) + remainder < (d + 1) * N);
      lattice->offset_[k * (d + 1) + remainder] =
        hash_table.find(key.data(), true);
      lattice->barycentric_[k * (d + 1) + remainder] = barycentric[remainder];

      // Gather the extent statistics of the lattice.
      for (int j = 0; j < d; ++j) {
        min_key[j] = (std::min)(key[j], min_key[j]);
        max_key[j] = (std::max)(key[j], max_key[j]);
      }
    }
  }

  // Find the Neighbors of each lattice point
  // Get the number of vertices in the lattice
  const int M = hash_table.size();
  lattice->M_ = M;

  // Gather some debug information.
  std::ostringstream extent_string;
  for (int i = 0; i < d; ++i) {
    extent_string << (max_key[i] - min_key[i]) << ", ";
  }
  // LOG(INFO) << "lattice size: " << M
  //           << ", samples: " << N
  //           << ", mean occupancy: " << static_cast<float>(N * (d+1)) /
  //           M
  //           << ", extent: " << extent_string.str();

  // Create the neighborhood structure
  // blur_neighbors (filter_size-1) x M_ row-major
  const int size = get_filter_size(lattice->neighborhood_size_, d);
  lattice->blur_neighbors_.resize((size - 1) * M);

  std::vector<boost::int16_t> start_key(d + 1);
  std::vector<boost::int16_t> walking_key(d + 1);

  //  extract (d+1) x M matrix of immediate neighbour indices row-major
  std::vector<int> immediate_neighbors((d + 1) * M);
  for (int i = 0; i < M; ++i) {
    const boost::int16_t* key = hash_table.getKey(i);
    for (int dim = 0; dim <= d; ++dim) {
      std::copy(key, key + d + 1, walking_key.begin());
      advance_in_dimension(dim, 1, &walking_key);
      immediate_neighbors[i + M * dim] =
        hash_table.find(walking_key.data(), false);
    }
  }
  assert(immediate_neighbors.size() == (M * (d + 1)));

  // Lattice traversal using immediate neighbour indices.
  LatticeApproximateTraversal traverse(lattice->neighborhood_size_, d,
                                       immediate_neighbors, M);
  for (int i = 0; i < M; ++i) {
    int* neighbors = &lattice->blur_neighbors_[i];
    int n          = -1;
    NeighborhoodCallback yield(M, neighbors, &n);
    traverse.go(i, yield);
    assert(n + 1 == size);
  }
}

boost::shared_ptr<PermutohedralReverse> Permutohedral::compute(
  const cublasHandle_t& handle,
  const float* filter,
  const float* in,
  int num_output,
  int group,
  int value_size,
  bool do_skip_blur,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  float* out) const {
  // Setup blur operation. This op will be returned to be able to compute the
  // gradient later.
  // TODO(mkiefel): probably move to some kind of constructor or init.
  boost::shared_ptr<PermutohedralReverse> reverse_operation(
    new PermutohedralReverse());

  reverse_operation->init(filter, num_output, group, value_size, do_skip_blur,
                          in_offset, out_offset, in_size, out_size, lattice_);
  reverse_operation->compute(handle, in, out);

  return reverse_operation;
}

boost::shared_ptr<PermutohedralReverse> Permutohedral::max_compute(
  const float* filter,
  const float* in,
  int value_size,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  float* out) {
  // Setup max operation. This op will be returned to be able to compute the
  // gradient later.
  // TODO(mkiefel): probably move to some kind of constructor or init.
  boost::shared_ptr<PermutohedralReverse> reverse_operation(
    new PermutohedralReverse());

  reverse_operation->init(filter, value_size, value_size, value_size, false,
                          in_offset, out_offset, in_size, out_size, lattice_);
  reverse_operation->max_compute(in, out);

  return reverse_operation;
}

/************************************************/
/***           PermutohedralReverse           ***/
/************************************************/

void PermutohedralReverse::max_reverse(const float* diff_in,
                                       float* diff_out_in) {
  // Blob<float> sliced_out;
  // sliced_out->Reshape(1, 1, num_output_, M_);
  at::Tensor sliced_out =
    at::zeros({1, 1, num_output_, M_}, torch::CUDA(at::kFloat));
  slice_tick(diff_in, &sliced_out);

  // Blob<float> blurred_out;
  // maxxed_out->Reshape(1, 1, value_size_, M_ + 1);
  at::Tensor blurred_out =
    at::zeros({1, 1, value_size_, M_ + 1}, torch::CUDA(at::kFloat));
  max_tick(sliced_out, &blurred_out);

  // Blob<float> splatted_out;
  splat_tick(blurred_out, diff_out_in);
}

PermutohedralReverse::PermutohedralReverse() {}

void PermutohedralReverse::init(
  const float* filter,
  int num_output,
  int group,
  int value_size,
  bool do_skip_blur,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  const boost::shared_ptr<const Permutohedral::Lattice> lattice) {
  d_                 = lattice->d_;
  N_                 = lattice->N_;
  neighborhood_size_ = lattice->neighborhood_size_;
  M_                 = lattice->M_;

  const int size = get_filter_size(neighborhood_size_, d_);

  // Get the lattice data over to the card.
  // TODO(mkiefel): this should be done in the lattice structure itself so it
  // can be actually be reused between computation calls.
  // barycentric_.Reshape(1, 1, N_, d_ + 1);
  barycentric_ = at::zeros({1, 1, N_, d_ + 1}, torch::CUDA(at::kFloat));
  assert(lattice->barycentric_.size() == N_ * (d_ + 1));
  std::copy(lattice->barycentric_.begin(), lattice->barycentric_.end(),
            barycentric_.data<float>());

  // offset_.Reshape(1, 1, N_, d_ + 1);
  offset_ = at::zeros({1, 1, N_, d_ + 1}, torch::CUDA(at::kInt));
  assert(lattice->offset_.size() == N_ * (d_ + 1));
  std::copy(lattice->offset_.begin(), lattice->offset_.end(),
            offset_.data<float>());

  if (size > 1) {
    // blur_neighbors_.Reshape(1, 1, size - 1, M_);
    blur_neighbors_ = at::zeros({1, 1, size - 1, M_}, torch::CUDA(at::kInt));
    assert(lattice->blur_neighbors_.size() == (size - 1) * M_);
    std::copy(lattice->blur_neighbors_.begin(), lattice->blur_neighbors_.end(),
              blur_neighbors_.data<float>());
  }

  // Set the rest of the metadata.
  in_offset_  = in_offset;
  out_offset_ = out_offset;
  in_size_    = in_size;
  out_size_   = out_size;

  // filter_.Reshape(1, num_output, value_size / group, size);
  filter_ = at::zeros({1, num_output, value_size / group, size},
                      torch::CUDA(at::kFloat));
  ::gpu_memcpy(filter_.numel() * sizeof(float), filter, filter_.data<float>());

  num_output_   = num_output;
  group_        = group;
  value_size_   = value_size;
  do_skip_blur_ = do_skip_blur;

  // splatted_.Reshape(1, 1, M_ + 1, value_size);
  // max_idx_.Reshape(1, 1, M_, value_size);
  splatted_ = at::zeros({1, 1, M_ + 1, value_size}, torch::CUDA(at::kFloat));
  max_idx_  = at::zeros({1, 1, M_, value_size}, torch::CUDA(at::kInt));
}

void PermutohedralReverse::max_compute(const float* in, float* out) {
  // Blob<float> blurred_(1, 1, M_, num_output_);
  at::Tensor blurred_ =
    at::zeros({1, 1, M_, num_output_}, torch::CUDA(at::kFloat));

  splat(in, &splatted_);

  max(splatted_, &blurred_);

  slice(blurred_, out);
}

void PermutohedralReverse::blur(const cublasHandle_t& handle,
                                const at::Tensor& splatted,
                                const at::Tensor& filter,
                                at::Tensor* blurred) const {
  // filter         num_output x value_size / group x filter_size row-major
  // splatted       value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // blurred        num_output x M_                               row-major
  const int size = get_filter_size(neighborhood_size_, d_);

  const int M = num_output_ / group_;
  const int K = value_size_ / group_ * size;
  const int N = M_;

  const int max_size   = 1024 * 1024 * 200;
  const int chunk_size = std::max<int>(1, std::min<int>(max_size / K, N));
  const int chunks     = std::ceil(static_cast<double>(N) / chunk_size);

  // Blob<float> col_data_blob(1, 1, K, chunk_size);
  at::Tensor col_data_blob =
    at::zeros({1, 1, K, chunk_size}, torch::CUDA(at::kFloat));

  // number of filter parameters in a group
  const int filter_offset = M * K;
  // number of values in an output region / column
  const int top_offset = M * N;

  const int* blur_neighbors = 0;
  if (size > 1) {
    blur_neighbors = blur_neighbors_.data<int>();
  }

  float* const col_data = col_data_blob.data<float>();

  for (int g = 0; g < group_; ++g) {
    for (int c = 0; c < chunks; ++c) {
      const int start = c * chunk_size;
      const int end   = std::min<int>(N, start + chunk_size);

      im2col(splatted.data<float>() + (value_size_ / group_) * g * (N + 1),
             value_size_ / group_, size, M_, start, end, blur_neighbors,
             col_data);

      ::gpu_gemm_ex(
        handle, CblasNoTrans, CblasNoTrans, M, end - start, K,
        static_cast<float>(1), filter.data<float>() + filter_offset * g, K,
        col_data, end - start, static_cast<float>(0),
        blurred->data<float>() + top_offset * g + chunk_size * c, N);
    }
  }
}

void PermutohedralReverse::blur_tick(const cublasHandle_t& handle,
                                     const at::Tensor& blurred_tick,
                                     at::Tensor* blurred_out,
                                     float* filter_out) {
  // filter_        num_output x value_size / group x filter_size row-major
  // blurred_out    value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // blurred_tick   num_output x M_                               row-major
  const int size = get_filter_size(neighborhood_size_, d_);

  const int M = num_output_ / group_;
  const int K = value_size_ / group_ * size;
  const int N = M_;

  const int max_size   = 1024 * 1024 * 200 / 2;
  const int chunk_size = std::max<int>(1, std::min<int>(max_size / K, N));
  const int chunks     = std::ceil(static_cast<double>(N) / chunk_size);

  // Blob<float> col_data_blob(1, 1, K, chunk_size);
  at::Tensor col_data_blob =
    at::zeros({1, 1, K, chunk_size}, torch::CUDA(at::kFloat));
  at::Tensor col_diff_blob =
    at::zeros({1, 1, K, chunk_size}, torch::CUDA(at::kFloat));

  // number of filter parameters in a group
  const int filter_offset = M * K;
  // number of values in an output region / column
  const int top_offset = M * N;

  const int* blur_neighbors = 0;
  if (size > 1) {
    blur_neighbors = blur_neighbors_.data<int>();
  }

  float* const col_data      = col_data_blob.data<float>();
  float* const col_diff_data = col_diff_blob.data<float>();

  for (int g = 0; g < group_; ++g) {
    for (int c = 0; c < chunks; ++c) {
      const int start = c * chunk_size;
      const int end   = std::min<int>(N, start + chunk_size);

      im2col(splatted_.data<float>() + (value_size_ / group_) * g * (N + 1),
             value_size_ / group_, size, M_, start, end, blur_neighbors,
             col_data);

      // Gradient w.r.t. filter.
      ::gpu_gemm_ex(
        handle, CblasNoTrans, CblasTrans, M, K, end - start,
        static_cast<float>(1),
        blurred_tick.data<float>() + top_offset * g + chunk_size * c, N,
        col_data, end - start, static_cast<float>(1),
        filter_out + filter_offset * g, K);

      // Gradient w.r.t. data.
      ::gpu_gemm_ex(
        handle, CblasTrans, CblasNoTrans, K, end - start, M,
        static_cast<float>(1), filter_.data<float>() + filter_offset * g, K,
        blurred_tick.data<float>() + top_offset * g + chunk_size * c, N,
        static_cast<float>(0), col_diff_data, end - start);

      col2im(col_diff_data, value_size_ / group_, size, M_, start, end,
             blur_neighbors,
             blurred_out->data<float>() + (value_size_ / group_) * g * (N + 1));
    }
  }
}

void PermutohedralReverse::reverse(const cublasHandle_t& handle,
                                   const float* diff_in,
                                   float* diff_out_filter,
                                   float* diff_out_in) {
  // Blob<float> sliced_out;
  // sliced_out->Reshape(1, 1, num_output_, M_);
  at::Tensor sliced_out =
    at::zeros({1, 1, num_output_, M_}, torch::CUDA(at::kFloat));
  slice_tick(diff_in, &sliced_out);

  // Blob<float> blurred_out;
  // blurred_out.Reshape(1, 1, value_size_, M_ + 1);
  at::Tensor blurred_out =
    at::zeros({1, 1, value_size_, M_ + 1}, torch::CUDA(at::kFloat));

  if (do_skip_blur_) {
    copy_to_blurred_out_data<<<CUDA_GET_BLOCKS(value_size_),
                               CUDA_NUM_THREADS>>>(
      sliced_out.data<float>(), M_, value_size_, blurred_out.data<float>());

    CUDA_POST_KERNEL_CHECK;
  } else {
    blur_tick(handle, sliced_out, &blurred_out, diff_out_filter);
  }

  // Blob<float> splatted_out;
  splat_tick(blurred_out, diff_out_in);
}

void PermutohedralReverse::max_tick(const at::Tensor& maxxed_tick,
                                    at::Tensor* maxxed_out) {
  // filter_       num_output x value_size / group x filter_size row-major
  // maxxed_out    value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                             row-major
  // maxxed_tick   value_size x M_                               row-major
  const float* tick_data = maxxed_tick.data<float>();

  const int filter_size = get_filter_size(neighborhood_size_, d_);

  const int* blur_neighbors = 0;
  if (filter_size > 1) {
    blur_neighbors = blur_neighbors_.data<int>();
  }

  float* max_out_tick_data = maxxed_out->data<float>();

  maxtick_gpu_kernel<<<CUDA_GET_BLOCKS(value_size_), CUDA_NUM_THREADS>>>(
    tick_data, value_size_, filter_size, M_, blur_neighbors, max_out_tick_data,
    max_idx_.data<int>());

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::compute(const cublasHandle_t& handle,
                                   const float* in,
                                   float* out) {
  // Blob<float> blurred_(1, 1, M_, num_output_);
  at::Tensor blurred_ =
    at::zeros({1, 1, M_, num_output_}, torch::CUDA(at::kFloat));

  splat(in, &splatted_);

  if (do_skip_blur_) {
    copy_to_blur_data<<<CUDA_GET_BLOCKS(num_output_), CUDA_NUM_THREADS>>>(
      splatted_.data<float>(), M_, num_output_, blurred_.data<float>());

    CUDA_POST_KERNEL_CHECK;
  } else {
    blur(handle, splatted_, filter_, &blurred_);
  }

  slice(blurred_, out);
}

void PermutohedralReverse::slice(const at::Tensor& data, float* sliced) const {
  // data           num_output x M_                               row-major
  // sliced         num_output x out_size                         row-major
  slice_gpu_kernel<<<CUDA_GET_BLOCKS(out_size_), CUDA_NUM_THREADS>>>(
    out_size_, data.data<float>(), M_, d_, out_offset_, num_output_,
    offset_.data<int>(), barycentric_.data<float>(), sliced);

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::splat(const float* in, at::Tensor* splatted) const {
  // in             value_size x in_size                          row-major
  // splatted       value_size x (M_+1)                           row-major
  ::gpu_memset(splatted->numel() * sizeof(float), 0, splatted->data<float>());
  splat_gpu_kernel<<<CUDA_GET_BLOCKS(in_size_), CUDA_NUM_THREADS>>>(
    in_size_, in, M_, d_, in_offset_, value_size_, offset_.data<int>(),
    barycentric_.data<float>(), splatted->data<float>());

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::im2col(const float* im,
                                  const int value_size,
                                  const int filter_size,
                                  const int M,
                                  const int start,
                                  const int end,
                                  const int* blur_neighbors,
                                  float* col) {
  // im             value_size      x (M_+1) row-major blur_neighbors
  // (filter_size-1) x M_                              row-major col
  // value_size x filter_size x (end - start)          row-major
  const int output_size = end - start;

  im2col_gpu_kernel<<<CUDA_GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
    im, value_size, filter_size, M, start, output_size, blur_neighbors, col);

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::col2im(const float* col,
                                  const int value_size,
                                  const int filter_size,
                                  const int M,
                                  const int start,
                                  const int end,
                                  const int* blur_neighbors,
                                  float* im) {
  // col            value_size x filter_size x (end - start) row-major
  // blur_neighbors (filter_size-1) x M_ row-major im             value_size
  // x (M_+1)                          row-major

  // ::gpu_memset(value_size * (M_ + 1) * sizeof(float), im, 0);

  const int output_size = end - start;

  col2im_gpu_kernel<<<CUDA_GET_BLOCKS(output_size), CUDA_NUM_THREADS>>>(
    col, value_size, filter_size, M, start, output_size, blur_neighbors, im);

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::slice_tick(const float* sliced_tick,
                                      at::Tensor* sliced_out) const {
  // sliced_tick        num_output x out_size row-major sliced_out num_output
  // x M_                               row-major
  slice_tick_gpu_kernel<<<CUDA_GET_BLOCKS(out_size_), CUDA_NUM_THREADS>>>(
    out_size_, sliced_tick, M_, d_, out_offset_, num_output_,
    offset_.data<int>(), barycentric_.data<float>(), sliced_out->data<float>());

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::max(const at::Tensor& splatted, at::Tensor* maxxed) {
  // splatted       value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // maxxed         num_output x M_                               row-major
  const int filter_size = get_filter_size(neighborhood_size_, d_);

  const float* splat_data = splatted.data<float>();
  float* max_data         = maxxed->data<float>();

  const int* blur_neighbors = 0;
  if (filter_size > 1) {
    blur_neighbors = blur_neighbors_.data<int>();
  }

  maxpool_gpu_kernel<<<CUDA_GET_BLOCKS(value_size_), CUDA_NUM_THREADS>>>(
    splat_data, value_size_, filter_size, M_, blur_neighbors, max_data,
    max_idx_.data<int>());

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::splat_tick(const at::Tensor& splatted_tick,
                                      float* splatted_out) {
  // Blob<float>& splatted_tick
  // splatted_tick  value_size x (M_+1)                           row-major
  // splatted_out   value_size x in_size                          row-major
  splat_tick_gpu_kernel<<<CUDA_GET_BLOCKS(in_size_), CUDA_NUM_THREADS>>>(
    in_size_, splatted_tick.data<float>(), M_, d_, in_offset_, value_size_,
    offset_.data<int>(), barycentric_.data<float>(), splatted_out);

  CUDA_POST_KERNEL_CHECK;
}
