/*
 * @Author: Haozhe Xie
 * @Date:   2019-09-03 09:28:46
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2019-09-04 16:01:45
 * @Email:  cshzxie@gmail.com
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <stdio.h>
#include <iostream>

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
/***               CUDA Vector                ***/
/************************************************/
template <typename T>
CUDAVector<T>::CUDAVector() : CUDAVector(32) {}

template <typename T>
CUDAVector<T>::CUDAVector(int capacity) {
  _capacity = capacity;
  _size     = 0;
  cudaMalloc((void**)&_data, _capacity * sizeof(T));
}

template <typename T>
CUDAVector<T>::~CUDAVector() {
  cudaFree(_data);
}

template <typename T>
void CUDAVector<T>::grow(int capacity) {
  _capacity =
    (capacity == -1 || capacity < _capacity) ? _capacity * 2 : capacity;

  T* old_data = _data;
  cudaMalloc((void**)&_data, _capacity * sizeof(T));

  for (int i = 0; i < _size; ++i) {
    _data[i] = old_data[i];
  }
  cudaFree(old_data);
}

template <typename T>
T* CUDAVector<T>::data() const {
  return _data;
}

template <typename T>
int CUDAVector<T>::size() const {
  return _size;
}

template <typename T>
void CUDAVector<T>::resize(int size) {
  if (size > _capacity) {
    grow(size);
  }
  _size = size;
}

template <typename T>
void CUDAVector<T>::push_back(T t) {
  if (_size == _capacity) {
    grow();
  }
  _data[_size++] = t;
}

template <typename T>
void CUDAVector<T>::swap(CUDAVector& cv) {
  std::swap(_capacity, cv._capacity);
  std::swap(_size, cv._size);
  std::swap(_data, cv._data);
}

template <typename T>
T CUDAVector<T>::pop_back() {
  --_size;
  return _data[_size];
}

template <typename T>
T* CUDAVector<T>::begin() {
  return _data;
}

template <typename T>
T* CUDAVector<T>::end() {
  return _data + _size;
}

template <typename T>
T& CUDAVector<T>::operator[](int i) {
  return _data[i];
}

template <typename T>
const T& CUDAVector<T>::operator[](int i) const {
  return _data[i];
}

template <typename U>
std::ostream& operator<<(std::ostream& os, const CUDAVector<U>& cv) {
  std::vector<U> vec(cv._size);
  ::gpu_memcpy(cv._size * sizeof(U), cv._data, vec.data());

  os << vec;
  return os;
}

/************************************************/
/*** Gaussian Filter for Permutohedral Lattice **/
/************************************************/

void GaussianFilter::build_filter(const cublasHandle_t& handle) {
  boost::array<float, 2> gauss = {{1, 0.5}};
  const int size = get_filter_size(neighborhood_size_, feature_size_);
  HashTable hash_table(feature_size_, size * (feature_size_ + 1));
  std::vector<float> lattice(size + 1);

  // Insert center of lattice into hash table.
  std::vector<boost::int16_t> center(feature_size_ + 1);
  const int center_index = hash_table.find(center.data(), true) + 1;
  assert(center_index == 1);

  // Insert all other lattice points into the hash table.
  LatticeTraversal traversal(neighborhood_size_, feature_size_);
  TraversalCallback yield(hash_table);
  traversal.go(center, yield);

  // Initialize the center of the lattice.
  lattice[center_index] = 1;

  std::vector<float> tmp_lattice(size + 1);
  std::vector<boost::int16_t> walking_key_up(feature_size_ + 1);
  std::vector<boost::int16_t> walking_key_down(feature_size_ + 1);
  for (int d = 0; d <= feature_size_; ++d) {
    std::fill(tmp_lattice.begin(), tmp_lattice.end(), 0);

    for (int i = 0; i < size; ++i) {
      const boost::int16_t* key = hash_table.get_key(i);
      std::copy(key, key + feature_size_ + 1, walking_key_up.begin());
      std::copy(key, key + feature_size_ + 1, walking_key_down.begin());

      float& v = tmp_lattice[i + 1];
      v        = lattice[i + 1] * gauss[0];

      for (int n = 1; n < neighborhood_size_ + 1; ++n) {
        advance_in_dimension(d, 1, &walking_key_up);
        advance_in_dimension(d, -1, &walking_key_down);

        v += (lattice[hash_table.find(walking_key_up.data()) + 1] +
              lattice[hash_table.find(walking_key_down.data()) + 1]) *
             (n < gauss.size() ? gauss[n] : 0);
      }
    }

    lattice.swap(tmp_lattice);
    lattice[0] = 0;
  }

  filter_.resize(size);
  // Normalize the filter according to the center lattice point. Like that we
  // are not creating additional energy for it.
  // const value_type alpha = lattice[1];
  // for (int i = 0; i < size; ++i) {
  //   filter_[i] = lattice[i + 1] / alpha;
  // }
  const float alpha = lattice[1];
  ::gpu_memcpy(size * sizeof(float), lattice.data() + 1, filter_.data());
  ::gpu_scal(handle, size, 1.0 / alpha, filter_.data());
}

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

void Permutohedral::map_back(const CUDAVector<boost::int16_t>& key,
                             float* const x) {
  const int d_      = key.size() - 1;
  float inv_std_dev = std::sqrt(2.0 / 3.0) * (d_ + 1);

  CUDAVector<float> scale_factor(d_);
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
  lattice_                    = lattice;
  lattice->N_                 = N;
  lattice->d_                 = d;
  lattice->neighborhood_size_ = neighborhood_size;

  // Allocate enough storage
  std::vector<boost::int16_t> ranks;
  ranks.resize((d + 1) * N);
  lattice->offset_.resize((d + 1) * N);
  lattice->barycentric_.resize((d + 1) * N);

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
  std::vector<boost::int16_t> min_key(d + 1);
  std::vector<boost::int16_t> max_key(d + 1);
  // lattice->offset_ CPU data
  std::vector<int> lattice_offset((d + 1) * N);
  // lattice->barycentric_ CPU data
  std::vector<float> lattice_barycentric((d + 1) * N);

  // Compute the canonical simplex
  for (int i = 0; i <= d; ++i) {
    for (int j = 0; j <= d - i; ++j) {
      canonical[i * (d + 1) + j] = i;
    }
    for (int j = d - i + 1; j <= d; ++j) {
      canonical[i * (d + 1) + j] = i - (d + 1);
    }
  }

  // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
  float inv_std_dev = sqrt(2.0 / 3.0) * (d + 1);
  // Compute the diagonal part of E (p.5 in [Adams etal 2010])
  for (int i = 0; i < d; ++i) {
    scale_factor[i] = 1.0 / sqrt((i + 2) * (i + 1)) * inv_std_dev;
  }

  // Compute the simplex each feature lies in
  for (int k = 0; k < N; ++k) {
    // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
    // const float * f = feature + k*feature_size;

    // sm contains the sum of 1..n of our faeture vector
    float sm(0);
    for (int j = d; j > 0; j--) {
      const int fIndex = (j - 1) * N + k;
      float cf         = feature[fIndex] * scale_factor[j - 1];
      elevated[j]      = sm - j * cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    float down_factor = 1.0f / (d + 1);
    float up_factor   = (d + 1);
    int sum           = 0;
    for (int i = 0; i <= d; ++i) {
      int rd  = round(down_factor * static_cast<float>(elevated[i]));
      rem0[i] = rd * up_factor;
      sum += rd;
    }

    // Find the simplex we are in and store it in rank (where rank
    // describes what position coorinate i has in the sorted order of the
    // features values)
    boost::int16_t* rank = ranks.data() + (d + 1) * k;
    for (int i = 0; i < d; ++i) {
      double di = static_cast<float>(elevated[i]) - rem0[i];
      for (int j = i + 1; j <= d; ++j)
        if (di < static_cast<float>(elevated[j]) - rem0[j]) {
          ++rank[i];
        } else {
          ++rank[j];
        }
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= d; ++i) {
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
      for (int i = 0; i <= d + 1; ++i) {
        barycentric[i] = 1.0;
      }
    } else {
      for (int i = 0; i <= d + 1; ++i) {
        barycentric[i] = 0;
      }
      for (int i = 0; i <= d; ++i) {
        float v = (elevated[i] - rem0[i]) * down_factor;

        if (d - rank[i] < 0 || d - rank[i] + 1 >= d + 2) {
          throw std::runtime_error("Permutohedral: rank access error");
        }
        // assert(d_-rank[i]   >= 0);
        // assert(d_-rank[i]+1 <  d_+2);
        barycentric[d - rank[i]] += v;
        barycentric[d - rank[i] + 1] -= v;
      }
      // Wrap around
      barycentric[0] += 1.0 + barycentric[d + 1];
    }

    // Compute all vertices and their offset
    // CUDAVector<boost::int16_t> neighborKeyUp(d + 1);
    // CUDAVector<boost::int16_t> neighborKeyDown(d + 1);
    for (int remainder = 0; remainder <= d; ++remainder) {
      for (int i = 0; i < d; ++i) {
        key[i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]];
      }
      assert(k * (d + 1) + remainder < (d + 1) * N);
      // lattice->offset_[k * (d + 1) + remainder] = hash_table.find(key.data(),
      // true);
      lattice_offset[k * (d + 1) + remainder] =
        hash_table.find(key.data(), true);
      // lattice->barycentric_[k * (d + 1) + remainder] =
      // barycentric[remainder];
      lattice_barycentric[k * (d + 1) + remainder] = barycentric[remainder];
      // Gather the extent statistics of the lattice.
      // for (int j = 0; j < d; ++j) {
      //   min_key[j] = (std::min)(key[j], min_key[j]);
      //   max_key[j] = (std::max)(key[j], max_key[j]);
      // }
    }
  }
  ::gpu_memcpy((d + 1) * N * sizeof(int), lattice_offset.data(),
               lattice->offset_.data());
  ::gpu_memcpy((d + 1) * N * sizeof(float), lattice_barycentric.data(),
               lattice->barycentric_.data());

  // Find the Neighbors of each lattice point
  // Get the number of vertices in the lattice
  const int M = hash_table.size();
  lattice->M_ = M;

  // Gather some debug information.
  // std::ostringstream extent_string;
  // for (int i = 0; i < d; ++i) {
  //   extent_string << (max_key[i] - min_key[i]) << ", ";
  // }
  // LOG(INFO) << "lattice size: " << M
  //           << ", samples: " << N
  //           << ", mean occupancy: " << static_cast<float>(N * (d+1)) /
  //           M
  //           << ", extent: " << extent_string.str();

  // Create the neighborhood structure
  // blur_neighbors (filter_size-1) x M_ row-major
  const int size = get_filter_size(lattice->neighborhood_size_, d);
  // lattice->blur_neighbors_
  std::vector<int> lattice_blur_neighbors((size - 1) * M);
  lattice->blur_neighbors_.resize((size - 1) * M);

  // CUDAVector<boost::int16_t> start_key(d + 1);
  std::vector<boost::int16_t> walking_key(d + 1);

  //  extract (d+1) x M matrix of immediate neighbour indices row-major
  std::vector<int> immediate_neighbors((d + 1) * M);
  for (int i = 0; i < M; ++i) {
    const boost::int16_t* key = hash_table.get_key(i);
    for (int dim = 0; dim <= d; ++dim) {
      std::copy(key, key + d + 1, walking_key.begin());
      // ::gpu_memcpy((d + 1) * sizeof(boost::int16_t), key,
      // walking_key.begin());

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
    int* neighbors = &lattice_blur_neighbors[i];
    int n          = -1;
    NeighborhoodCallback yield(M, neighbors, &n);
    traverse.go(i, yield);
    assert(n + 1 == size);
  }
  ::gpu_memcpy((size - 1) * M * sizeof(int), lattice_blur_neighbors.data(),
               lattice->blur_neighbors_.data());
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
  torch::Tensor sliced_out =
    torch::zeros({1, 1, num_output_, M_}, torch::CUDA(torch::kFloat));
  slice_tick(diff_in, &sliced_out);

  // Blob<float> blurred_out;
  // maxxed_out->Reshape(1, 1, value_size_, M_ + 1);
  torch::Tensor blurred_out =
    torch::zeros({1, 1, value_size_, M_ + 1}, torch::CUDA(torch::kFloat));
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
  barycentric_ = torch::zeros({1, 1, N_, d_ + 1}, torch::CUDA(torch::kFloat));
  assert(lattice->barycentric_.size() == N_ * (d_ + 1));
  // std::copy(lattice->barycentric_.begin(), lattice->barycentric_.end(),
  // barycentric_.data<float>());
  ::gpu_memcpy(lattice->barycentric_.size() * sizeof(float),
               lattice->barycentric_.data(), barycentric_.data<float>());

  // offset_.Reshape(1, 1, N_, d_ + 1);
  offset_ = torch::zeros({1, 1, N_, d_ + 1}, torch::CUDA(torch::kInt));
  assert(lattice->offset_.size() == N_ * (d_ + 1));
  // std::copy(lattice->offset_.begin(), lattice->offset_.end(),
  // offset_.data<float>());
  ::gpu_memcpy(lattice->offset_.size() * sizeof(int), lattice->offset_.data(),
               offset_.data<int>());

  if (size > 1) {
    // blur_neighbors_.Reshape(1, 1, size - 1, M_);
    blur_neighbors_ =
      torch::zeros({1, 1, size - 1, M_}, torch::CUDA(torch::kInt));
    assert(lattice->blur_neighbors_.size() == (size - 1) * M_);
    // std::copy(lattice->blur_neighbors_.begin(),
    // lattice->blur_neighbors_.end(), blur_neighbors_.data<float>());
    ::gpu_memcpy(lattice->blur_neighbors_.size() * sizeof(int),
                 lattice->blur_neighbors_.data(), blur_neighbors_.data<int>());
  }

  // Set the rest of the metadata.
  in_offset_  = in_offset;
  out_offset_ = out_offset;
  in_size_    = in_size;
  out_size_   = out_size;

  // filter_.Reshape(1, num_output, value_size / group, size);
  filter_ = torch::zeros({1, num_output, value_size / group, size},
                         torch::CUDA(torch::kFloat));
  ::gpu_memcpy(filter_.numel() * sizeof(float), filter, filter_.data<float>());

  num_output_   = num_output;
  group_        = group;
  value_size_   = value_size;
  do_skip_blur_ = do_skip_blur;

  // splatted_.Reshape(1, 1, M_ + 1, value_size);
  // max_idx_.Reshape(1, 1, M_, value_size);
  splatted_ =
    torch::zeros({1, 1, M_ + 1, value_size}, torch::CUDA(torch::kFloat));
  max_idx_ = torch::zeros({1, 1, M_, value_size}, torch::CUDA(torch::kInt));
}

void PermutohedralReverse::max_compute(const float* in, float* out) {
  // Blob<float> blurred_(1, 1, M_, num_output_);
  torch::Tensor blurred_ =
    torch::zeros({1, 1, M_, num_output_}, torch::CUDA(torch::kFloat));

  splat(in, &splatted_);

  max(splatted_, &blurred_);

  slice(blurred_, out);
}

void PermutohedralReverse::blur(const cublasHandle_t& handle,
                                const torch::Tensor& splatted,
                                const torch::Tensor& filter,
                                torch::Tensor* blurred) const {
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
  torch::Tensor col_data_blob =
    torch::zeros({1, 1, K, chunk_size}, torch::CUDA(torch::kFloat));

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
                                     const torch::Tensor& blurred_tick,
                                     torch::Tensor* blurred_out,
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
  torch::Tensor col_data_blob =
    torch::zeros({1, 1, K, chunk_size}, torch::CUDA(torch::kFloat));
  torch::Tensor col_diff_blob =
    torch::zeros({1, 1, K, chunk_size}, torch::CUDA(torch::kFloat));

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
  torch::Tensor sliced_out =
    torch::zeros({1, 1, num_output_, M_}, torch::CUDA(torch::kFloat));
  slice_tick(diff_in, &sliced_out);

  // Blob<float> blurred_out;
  // blurred_out.Reshape(1, 1, value_size_, M_ + 1);
  torch::Tensor blurred_out =
    torch::zeros({1, 1, value_size_, M_ + 1}, torch::CUDA(torch::kFloat));

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

void PermutohedralReverse::max_tick(const torch::Tensor& maxxed_tick,
                                    torch::Tensor* maxxed_out) {
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
  torch::Tensor blurred_ =
    torch::zeros({1, 1, M_, num_output_}, torch::CUDA(torch::kFloat));

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

void PermutohedralReverse::slice(const torch::Tensor& data,
                                 float* sliced) const {
  // data           num_output x M_                               row-major
  // sliced         num_output x out_size                         row-major
  slice_gpu_kernel<<<CUDA_GET_BLOCKS(out_size_), CUDA_NUM_THREADS>>>(
    out_size_, data.data<float>(), M_, d_, out_offset_, num_output_,
    offset_.data<int>(), barycentric_.data<float>(), sliced);

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::splat(const float* in,
                                 torch::Tensor* splatted) const {
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
                                      torch::Tensor* sliced_out) const {
  // sliced_tick        num_output x out_size row-major sliced_out num_output
  // x M_                               row-major
  slice_tick_gpu_kernel<<<CUDA_GET_BLOCKS(out_size_), CUDA_NUM_THREADS>>>(
    out_size_, sliced_tick, M_, d_, out_offset_, num_output_,
    offset_.data<int>(), barycentric_.data<float>(), sliced_out->data<float>());

  CUDA_POST_KERNEL_CHECK;
}

void PermutohedralReverse::max(const torch::Tensor& splatted,
                               torch::Tensor* maxxed) {
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

void PermutohedralReverse::splat_tick(const torch::Tensor& splatted_tick,
                                      float* splatted_out) {
  // Blob<float>& splatted_tick
  // splatted_tick  value_size x (M_+1)                           row-major
  // splatted_out   value_size x in_size                          row-major
  splat_tick_gpu_kernel<<<CUDA_GET_BLOCKS(in_size_), CUDA_NUM_THREADS>>>(
    in_size_, splatted_tick.data<float>(), M_, d_, in_offset_, value_size_,
    offset_.data<int>(), barycentric_.data<float>(), splatted_out);

  CUDA_POST_KERNEL_CHECK;
}

boost::shared_ptr<std::vector<BlurOperation>> init_lattice(
  const cublasHandle_t& handle,
  torch::Tensor& in_features,
  torch::Tensor& out_features,
  int neighborhood_size,
  int feature_size,
  bool do_visualization = false) {
  int batch_size = in_features.size(0);
  int in_height  = in_features.size(2);
  int in_width   = in_features.size(3);
  int out_height = out_features.size(2);
  int out_width  = out_features.size(3);
  int in_size    = in_height * in_width;
  int out_size   = out_height * out_width;
  int in_offset  = 0;
  int out_offset = in_size;
  int data_count = in_size + out_size;

  GaussianFilter gauss(neighborhood_size, feature_size);
  gauss.build_filter(handle);
  const float* gauss_filter = gauss.filter();

  boost::shared_ptr<std::vector<BlurOperation>> operations =
    boost::make_shared<std::vector<BlurOperation>>(batch_size);
  torch::Tensor features =
    torch::zeros({feature_size * data_count}, torch::CUDA(torch::kFloat));

  float* p_features     = features.data<float>();
  float* p_in_features  = in_features.data<float>();
  float* p_out_features = out_features.data<float>();

  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < feature_size; ++c) {
      int in_offset  = i * feature_size * in_size + c * in_size;
      int out_offset = i * feature_size * out_size + c * out_size;
      ::gpu_memcpy(in_size * sizeof(float), p_in_features + in_offset,
                   p_features + c * data_count);
      ::gpu_memcpy(out_size * sizeof(float), p_out_features + out_offset,
                   p_features + c * data_count + in_size);
    }

    BlurOperation& op = (*operations)[i];
    op.blur_.reset(new Permutohedral());

    op.blur_->init(features.cpu().data<float>(), data_count, feature_size,
                   neighborhood_size, do_visualization);
    op.norm_there_ =
      torch::ones({1, 1, in_height, in_width}, torch::CUDA(torch::kFloat));
    op.norm_back_ =
      torch::zeros({1, 1, out_height, out_width}, torch::CUDA(torch::kFloat));

    // Note (Haozhe Xie): NormType == AFTER, NormType == SYMMETRIC is not
    // supported yet!
    op.blur_->compute(handle, gauss_filter, op.norm_there_.data<float>(), 1, 1,
                      1, false, in_offset, out_offset, in_size, out_size,
                      op.norm_back_.data<float>());

    // for (int i = 0; i < op.norm_back_->count(); ++i) {
    //   norm_back_data[i] = 1.0 / (norm_back_data[i] + 1e-20);
    // }
    ::gpu_mul_inverse(out_size, op.norm_back_.data<float>(),
                      op.norm_back_.data<float>(), 1e-20);
  }
  return operations;
}

std::vector<torch::Tensor> permutohedral_cuda_forward(
  const cublasHandle_t& handle,
  int neighborhood_size,
  int group,
  int out_channels,
  bool do_skip_blur,
  torch::Tensor data,
  torch::Tensor in_features,
  torch::Tensor out_features,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor bias_multiplier) {
  int batch_size    = data.size(0);
  int data_channels = data.size(1);
  int in_channels   = in_features.size(1);
  int in_height     = in_features.size(2);
  int in_width      = in_features.size(3);
  int out_height    = out_features.size(2);
  int out_width     = out_features.size(3);
  int in_size       = in_height * in_width;
  int out_size      = out_height * out_width;
  int in_offset     = 0;
  int out_offset    = in_size;

  torch::Tensor output =
    torch::zeros({batch_size, out_channels, out_height, out_width},
                 torch::CUDA(torch::kFloat));

  boost::shared_ptr<std::vector<BlurOperation>> blur_operations = init_lattice(
    handle, in_features, out_features, neighborhood_size, in_channels);

  // Note (Haozhe Xie): OffsetType == None, OffsetType == FULL and OffsetType ==
  // DIAG are not supported yet!
  // boost::shared_ptr<torch::Tensor> shifted_filter
  // = get_offset_filter(weights);

  torch::Tensor scaled_there = torch::zeros(
    {1, data_channels, in_height, in_width}, torch::CUDA(torch::kFloat));
  for (int i = 0; i < batch_size; ++i) {
    BlurOperation& op = (*blur_operations)[i];
    for (int c = 0; c < data_channels; ++c) {
      ::gpu_mul(in_size, op.norm_there_.data<float>(),
                data.data<float>() + i * data_channels * in_size + c * in_size,
                scaled_there.data<float>() + c * in_size);
    }

    // Compute the permutohedral filter response
    op.reverse_ = op.blur_->compute(
      handle, weights.data<float>(), scaled_there.data<float>(), out_channels,
      group, data_channels, do_skip_blur, in_offset, out_offset, in_size,
      out_size, output.data<float>() + i * out_channels * out_size);

    for (int c = 0; c < out_channels; ++c) {
      ::gpu_mul(
        out_size, op.norm_back_.data<float>(),
        output.data<float>() + i * out_channels * out_size + c * out_size,
        output.data<float>() + i * out_channels * out_size + c * out_size);
    }

    // Add bias
    if (bias.size(0) && bias_multiplier.size(0)) {
      ::gpu_gemm(handle, CblasNoTrans, CblasNoTrans, out_channels, out_size, 1,
                 static_cast<float>(1), bias.data<float>(),
                 bias_multiplier.data<float>(), static_cast<float>(1.0),
                 output.data<float>() + i * out_channels * out_size);
    }
  }

  CUDA_POST_KERNEL_CHECK;

  // Collect tensors for backward propagation
  std::vector<torch::Tensor> saved_tensors;
  saved_tensors.push_back(output);
  for (int i = 0; i < batch_size; ++i) {
    BlurOperation& op = (*blur_operations)[i];
    saved_tensors.push_back(op.norm_there_);
    saved_tensors.push_back(op.norm_back_);
    saved_tensors.push_back(op.reverse_->get_constants());
    saved_tensors.push_back(op.reverse_->get_filter());
    saved_tensors.push_back(op.reverse_->get_splatted());
    saved_tensors.push_back(op.reverse_->get_offset());
    saved_tensors.push_back(op.reverse_->get_max_idx());
    saved_tensors.push_back(op.reverse_->get_barycentric());
    saved_tensors.push_back(op.reverse_->get_blur_neighbors());
  }
  return saved_tensors;
}

std::vector<torch::Tensor> permutohedral_cuda_backward(
  const cublasHandle_t& handle,
  const torch::Tensor bias_multiplier,
  const torch::Tensor grad_output,
  torch::Tensor grad_weights,
  torch::Tensor grad_bias,
  torch::Tensor grad_data,
  std::vector<torch::Tensor> saved_tensors) {
  int batch_size    = grad_data.size(0);
  int data_channels = grad_data.size(1);
  int in_height     = grad_data.size(2);
  int in_width      = grad_data.size(3);
  int out_channels  = grad_output.size(1);
  int out_height    = grad_output.size(2);
  int out_width     = grad_output.size(3);
  int in_size       = in_height * in_width;
  int out_size      = out_height * out_width;

  const int N_TENSORS = 9;
  std::vector<torch::Tensor> norm_there;
  std::vector<torch::Tensor> norm_back;
  std::vector<torch::Tensor> constants;
  std::vector<torch::Tensor> filter;
  std::vector<torch::Tensor> splatted;
  std::vector<torch::Tensor> offset;
  std::vector<torch::Tensor> max_idx;
  std::vector<torch::Tensor> barycentric;
  std::vector<torch::Tensor> blur_neighbors;
  for (int i = 0; i < batch_size; ++i) {
    norm_there.push_back(saved_tensors[i * N_TENSORS + 0]);
    norm_back.push_back(saved_tensors[i * N_TENSORS + 1]);
    constants.push_back(saved_tensors[i * N_TENSORS + 2]);
    filter.push_back(saved_tensors[i * N_TENSORS + 3]);
    splatted.push_back(saved_tensors[i * N_TENSORS + 4]);
    offset.push_back(saved_tensors[i * N_TENSORS + 5]);
    max_idx.push_back(saved_tensors[i * N_TENSORS + 6]);
    barycentric.push_back(saved_tensors[i * N_TENSORS + 7]);
    blur_neighbors.push_back(saved_tensors[i * N_TENSORS + 8]);
  }

  // Gradient with respect to bias
  if (grad_bias.size(0) && bias_multiplier.size(0)) {
    for (int i = 0; i < batch_size; ++i) {
      ::gpu_gemv(handle, CblasNoTrans, out_channels, out_size, 1.,
                 grad_output.data<float>() + i * out_channels * out_size,
                 bias_multiplier.data<float>(), 1., grad_bias.data<float>());
    }
  }

  // Gradient computation
  torch::Tensor scaled_back = torch::zeros(
    {1, out_channels, out_height, out_width}, torch::CUDA(torch::kFloat));

  for (int i = 0; i < batch_size; ++i) {
    const torch::Tensor _norm_there = norm_there.at(i);
    const torch::Tensor _norm_back  = norm_back.at(i);

    for (int c = 0; c < out_channels; ++c) {
      ::gpu_mul(
        out_size,
        grad_output.data<float>() + i * out_channels * out_size + c * out_size,
        _norm_back.data<float>(), scaled_back.data<float>() + c * out_size);
    }

    PermutohedralReverse reverse_(constants.at(i), filter.at(i), splatted.at(i),
                                  max_idx.at(i), barycentric.at(i),
                                  offset.at(i), blur_neighbors.at(i));
    reverse_.reverse(handle, scaled_back.data<float>(),
                     grad_weights.data<float>(),
                     grad_data.data<float>() + i * data_channels * in_size);

    for (int c = 0; c < data_channels; ++c) {
      ::gpu_mul(
        in_size, _norm_there.data<float>(),
        grad_data.data<float>() + i * data_channels * in_size + c * in_size,
        grad_data.data<float>() + i * data_channels * in_size + c * in_size);
    }
  }

  CUDA_POST_KERNEL_CHECK;
  return {grad_weights, grad_bias, grad_data};
}