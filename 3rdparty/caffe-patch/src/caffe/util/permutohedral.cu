// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include "caffe/util/permutohedral.hpp"

#include <algorithm>
#include <cassert>

#include "caffe/blob.hpp"
#include "caffe/common.cuh"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/new_math_utils.hpp"

namespace caffe {

namespace permutohedral {

/************************************************/
/***         PermutohedralReverseGpu          ***/
/************************************************/

template <typename T>
void PermutohedralReverseGpu<T>::init(
  const value_type* filter,
  int num_output,
  int group,
  int value_size,
  bool do_skip_blur,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  const boost::shared_ptr<const typename Permutohedral<value_type>::Lattice>
    lattice) {
  d_                 = lattice->d_;
  N_                 = lattice->N_;
  neighborhood_size_ = lattice->neighborhood_size_;
  M_                 = lattice->M_;

  const int size = get_filter_size(neighborhood_size_, d_);

  // Get the lattice data over to the card.
  // TODO(mkiefel): this should be done in the lattice structure itself so it
  // can be actually be reused between computation calls.
  barycentric_.Reshape(1, 1, N_, d_ + 1);
  assert(lattice->barycentric_.size() == N_ * (d_ + 1));
  std::copy(lattice->barycentric_.begin(), lattice->barycentric_.end(),
            barycentric_.mutable_cpu_data());

  offset_.Reshape(1, 1, N_, d_ + 1);
  assert(lattice->offset_.size() == N_ * (d_ + 1));
  std::copy(lattice->offset_.begin(), lattice->offset_.end(),
            offset_.mutable_cpu_data());

  if (size > 1) {
    blur_neighbors_.Reshape(1, 1, size - 1, M_);
    assert(lattice->blur_neighbors_.size() == (size - 1) * M_);
    std::copy(lattice->blur_neighbors_.begin(), lattice->blur_neighbors_.end(),
              blur_neighbors_.mutable_cpu_data());
  }

  // Set the rest of the metadata.
  in_offset_  = in_offset;
  out_offset_ = out_offset;
  in_size_    = in_size;
  out_size_   = out_size;

  filter_.Reshape(1, num_output, value_size / group, size);
  caffe_gpu_memcpy(filter_.count() * sizeof(value_type), filter,
                   filter_.mutable_gpu_data());

  num_output_ = num_output;
  group_      = group;
  value_size_ = value_size;

  do_skip_blur_ = do_skip_blur;

  splatted_.Reshape(1, 1, M_ + 1, value_size);

  max_idx_.Reshape(1, 1, M_, value_size);
}

template <typename T>
__global__ void copy_to_blur_data(const T* splat_data,
                                  const int M,
                                  const int num_output,
                                  T* blur_data) {
  CUDA_KERNEL_LOOP(k, num_output) {
    for (int t = 0; t < M; t++) {
      blur_data[k * M + t] = splat_data[k * (M + 1) + t + 1];
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::compute(const value_type* in,
                                         value_type* out) {
  Blob<value_type> blurred_(1, 1, M_, num_output_);

  splat(in, &splatted_);

  if (do_skip_blur_) {
    copy_to_blur_data<value_type>
      // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(num_output_), CAFFE_CUDA_NUM_THREADS>>>(
        splatted_.gpu_data(), M_, num_output_, blurred_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
  } else {
    blur(splatted_, filter_, &blurred_);
  }

  slice(blurred_, out);
}

template <typename T>
void PermutohedralReverseGpu<T>::max_compute(const value_type* in,
                                             value_type* out) {
  Blob<value_type> blurred_(1, 1, M_, num_output_);

  splat(in, &splatted_);

  max(splatted_, &blurred_);

  slice(blurred_, out);
}

template <typename T>
__global__ void slice_gpu_kernel(const int out_size,
                                 const T* data,
                                 const int M,
                                 const int d,
                                 const int out_offset,
                                 const int num_output,
                                 const int* offset,
                                 const T* barycentric,
                                 T* sliced) {
  typedef T value_type;

  CUDA_KERNEL_LOOP(i, out_size) {
    for (int k = 0; k < num_output; k++) sliced[k * out_size + i] = 0;

    for (int j = 0; j <= d; j++) {
      int o        = offset[(out_offset + i) * (d + 1) + j];
      value_type w = barycentric[(out_offset + i) * (d + 1) + j];

      for (int k = 0; k < num_output; k++) {
        sliced[k * out_size + i] += w * data[k * M + o];
      }
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::slice(const Blob<value_type>& data,
                                       value_type* sliced) const {
  // data           num_output x M_                               row-major
  // sliced         num_output x out_size                         row-major
  slice_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(out_size_), CAFFE_CUDA_NUM_THREADS>>>(
      out_size_, data.gpu_data(), M_, d_, out_offset_, num_output_,
      offset_.gpu_data(), barycentric_.gpu_data(), sliced);
  // sliced->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void im2col_gpu_kernel(const T* im,
                                  const int value_size,
                                  const int filter_size,
                                  const int M,
                                  const int start,
                                  const int output_size,
                                  const int* blur_neighbors,
                                  T* col) {
  CUDA_KERNEL_LOOP(i, output_size) {
    for (std::size_t k = 0; k < value_size; ++k) {
      col[(k * filter_size + 0) * output_size + i] =
        im[k * (M + 1) + (i + start + 1)];

      for (std::size_t f = 1; f < filter_size; ++f) {
        const int* neighbors = &blur_neighbors[(f - 1) * M + 0];

        col[(k * filter_size + f) * output_size + i] =
          im[k * (M + 1) + (neighbors[i + start] + 1)];
      }
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::im2col(const value_type* im,
                                        const std::size_t value_size,
                                        const std::size_t filter_size,
                                        const std::size_t M,
                                        const std::size_t start,
                                        const std::size_t end,
                                        const int* blur_neighbors,
                                        value_type* col) {
  // im             value_size      x (M_+1)                          row-major
  // blur_neighbors (filter_size-1) x M_                              row-major
  // col            value_size x filter_size x (end - start)          row-major

  const int output_size = end - start;

  im2col_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS>>>(
      im, value_size, filter_size, M, start, output_size, blur_neighbors, col);

  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void col2im_gpu_kernel(const T* col,
                                  const int value_size,
                                  const int filter_size,
                                  const int M,
                                  const int start,
                                  const int output_size,
                                  const int* blur_neighbors,
                                  T* im) {
  CUDA_KERNEL_LOOP(i, output_size) {
    for (std::size_t k = 0; k < value_size; ++k) {
      im[k * (M + 1) + (i + start + 1)] +=
        col[(k * filter_size + 0) * output_size + i];

      for (std::size_t f = 1; f < filter_size; ++f) {
        const int* neighbors = &blur_neighbors[(f - 1) * M + 0];

        im[k * (M + 1) + (neighbors[i + start] + 1)] +=
          col[(k * filter_size + f) * output_size + i];
      }
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::col2im(const value_type* col,
                                        const std::size_t value_size,
                                        const std::size_t filter_size,
                                        const std::size_t M,
                                        const std::size_t start,
                                        const std::size_t end,
                                        const int* blur_neighbors,
                                        value_type* im) {
  // col            value_size x filter_size x (end - start)          row-major
  // blur_neighbors (filter_size-1) x M_                              row-major
  // im             value_size      x (M_+1)                          row-major

  // caffe_gpu_memset(value_size * (M_ + 1) * sizeof(value_type), im, 0);

  const int output_size = end - start;

  col2im_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS>>>(
      col, value_size, filter_size, M, start, output_size, blur_neighbors, im);

  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
void PermutohedralReverseGpu<T>::blur(const Blob<value_type>& splatted,
                                      const Blob<value_type>& filter,
                                      Blob<value_type>* blurred) const {
  // filter         num_output x value_size / group x filter_size row-major
  // splatted       value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // blurred        num_output x M_                               row-major

  const int size = get_filter_size(neighborhood_size_, d_);

  const std::size_t M = num_output_ / group_;
  const std::size_t K = value_size_ / group_ * size;
  const std::size_t N = M_;

  const std::size_t max_size = 1024 * 1024 * 200;
  const std::size_t chunk_size =
    std::max<std::size_t>(1, std::min<std::size_t>(max_size / K, N));
  const std::size_t chunks = std::ceil(static_cast<double>(N) / chunk_size);

  Blob<value_type> col_data_blob(1, 1, K, chunk_size);

  // number of filter parameters in a group
  const std::size_t filter_offset = M * K;
  // number of values in an output region / column
  const std::size_t top_offset = M * N;

  const int* blur_neighbors = 0;
  if (size > 1) blur_neighbors = blur_neighbors_.gpu_data();

  value_type* const col_data = col_data_blob.mutable_gpu_data();

  for (std::size_t g = 0; g < group_; ++g) {
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::size_t start = c * chunk_size;
      const std::size_t end   = std::min<std::size_t>(N, start + chunk_size);

      im2col(splatted.gpu_data() + (value_size_ / group_) * g * (N + 1),
             value_size_ / group_, size, M_, start, end, blur_neighbors,
             col_data);

      caffe_gpu_gemm_ex<value_type>(
        CblasNoTrans, CblasNoTrans, M, end - start, K,
        static_cast<value_type>(1), filter.gpu_data() + filter_offset * g, K,
        col_data, end - start, static_cast<value_type>(0),
        blurred->mutable_gpu_data() + top_offset * g + chunk_size * c, N);
    }
  }
}

template <typename T>
__global__ void maxpool_gpu_kernel(const T* splat_data,
                                   const int value_size,
                                   const int filter_size,
                                   const int M,
                                   const int* blur_neighbors,
                                   T* maxxed_data,
                                   int* idx_data) {
  CUDA_KERNEL_LOOP(i, value_size) {
    for (std::size_t j = 0; j < M; ++j) {
      int idx     = j;
      T max_value = splat_data[i * (M + 1) + idx + 1];
      for (std::size_t k = 1; k < filter_size; ++k) {
        const int* neighbors = blur_neighbors + ((k - 1) * M);
        T value              = splat_data[i * (M + 1) + neighbors[j] + 1];
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

template <typename T>
void PermutohedralReverseGpu<T>::max(const Blob<value_type>& splatted,
                                     Blob<value_type>* maxxed) {
  // splatted       value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // maxxed         num_output x M_                               row-major

  const int filter_size = get_filter_size(neighborhood_size_, d_);

  const value_type* splat_data = splatted.gpu_data();
  value_type* max_data         = maxxed->mutable_gpu_data();

  const int* blur_neighbors = 0;
  if (filter_size > 1) blur_neighbors = blur_neighbors_.gpu_data();

  maxpool_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(value_size_), CAFFE_CUDA_NUM_THREADS>>>(
      splat_data, value_size_, filter_size, M_, blur_neighbors, max_data,
      max_idx_.mutable_gpu_data());

  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void splat_gpu_kernel(const int in_size,
                                 const T* in,
                                 const int M,
                                 const int d,
                                 const int in_offset,
                                 const int value_size,
                                 const int* offset,
                                 const T* barycentric,
                                 T* splatted) {
  typedef T value_type;

  CUDA_KERNEL_LOOP(i, in_size) {
    for (int j = 0; j <= d; ++j) {
      int o               = offset[(in_offset + i) * (d + 1) + j] + 1;
      const value_type& w = barycentric[(in_offset + i) * (d + 1) + j];

      for (int k = 0; k < value_size; k++) {
        atomicAdd(&splatted[k * (M + 1) + o], w * in[k * in_size + i]);
      }
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::splat(const value_type* in,
                                       Blob<value_type>* splatted) const {
  // in             value_size x in_size                          row-major
  // splatted       value_size x (M_+1)                           row-major
  caffe_gpu_memset(splatted->count() * sizeof(value_type), 0,
                   splatted->mutable_gpu_data());
  splat_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(in_size_), CAFFE_CUDA_NUM_THREADS>>>(
      in_size_, in, M_, d_, in_offset_, value_size_, offset_.gpu_data(),
      barycentric_.gpu_data(), splatted->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void copy_to_blurred_out_data(const T* sliced_out,
                                         const int M,
                                         const int value_size,
                                         T* blurred_out) {
  CUDA_KERNEL_LOOP(k, value_size) {
    for (int t = 0; t < M; t++) {
      blurred_out[k * (M + 1) + t + 1] += sliced_out[k * M + t];
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::reverse(const value_type* diff_in,
                                         value_type* diff_out_filter,
                                         value_type* diff_out_in) {
  Blob<value_type> sliced_out;

  slice_tick(diff_in, &sliced_out);

  Blob<value_type> blurred_out;

  if (do_skip_blur_) {
    blurred_out.Reshape(1, 1, value_size_, M_ + 1);
    caffe_gpu_memset(value_size_ * (M_ + 1) * sizeof(value_type), 0,
                     blurred_out.mutable_gpu_data());
    copy_to_blurred_out_data<value_type>
      // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(value_size_), CAFFE_CUDA_NUM_THREADS>>>(
        sliced_out.gpu_data(), M_, value_size_, blurred_out.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
  } else {
    blur_tick(sliced_out, &blurred_out, diff_out_filter);
  }

  Blob<value_type> splatted_out;
  splat_tick(blurred_out, diff_out_in);
}

template <typename T>
void PermutohedralReverseGpu<T>::max_reverse(const value_type* diff_in,
                                             value_type* diff_out_in) {
  Blob<value_type> sliced_out;

  slice_tick(diff_in, &sliced_out);

  Blob<value_type> blurred_out;
  max_tick(sliced_out, &blurred_out);

  Blob<value_type> splatted_out;
  splat_tick(blurred_out, diff_out_in);
}

template <typename T>
__global__ void slice_tick_gpu_kernel(const int out_size,
                                      const T* sliced,
                                      const int M,
                                      const int d,
                                      const int out_offset,
                                      const int num_output,
                                      const int* offset,
                                      const T* barycentric,
                                      T* data) {
  typedef T value_type;

  CUDA_KERNEL_LOOP(i, out_size) {
    for (int j = 0; j <= d; j++) {
      int o        = offset[(out_offset + i) * (d + 1) + j];
      value_type w = barycentric[(out_offset + i) * (d + 1) + j];

      for (int k = 0; k < num_output; k++) {
        atomicAdd(&data[k * M + o], w * sliced[k * out_size + i]);
      }
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::slice_tick(
  const value_type* sliced_tick, Blob<value_type>* sliced_out) const {
  // sliced_tick        num_output x out_size                         row-major
  // sliced_out         num_output x M_                               row-major
  sliced_out->Reshape(1, 1, num_output_, M_);

  slice_tick_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(out_size_), CAFFE_CUDA_NUM_THREADS>>>(
      out_size_, sliced_tick, M_, d_, out_offset_, num_output_,
      offset_.gpu_data(), barycentric_.gpu_data(),
      sliced_out->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
void PermutohedralReverseGpu<T>::blur_tick(const Blob<value_type>& blurred_tick,
                                           Blob<value_type>* blurred_out,
                                           value_type* filter_out) {
  // filter_        num_output x value_size / group x filter_size row-major
  // blurred_out    value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // blurred_tick   num_output x M_                               row-major
  blurred_out->Reshape(1, 1, value_size_, M_ + 1);
  caffe_gpu_memset(value_size_ * (M_ + 1) * sizeof(value_type), 0,
                   blurred_out->mutable_gpu_data());

  const int size = get_filter_size(neighborhood_size_, d_);

  const std::size_t M = num_output_ / group_;
  const std::size_t K = value_size_ / group_ * size;
  const std::size_t N = M_;

  const std::size_t max_size = 1024 * 1024 * 200 / 2;
  const std::size_t chunk_size =
    std::max<std::size_t>(1, std::min<std::size_t>(max_size / K, N));
  const std::size_t chunks = std::ceil(static_cast<double>(N) / chunk_size);

  Blob<value_type> col_data_blob(1, 1, K, chunk_size);

  // number of filter parameters in a group
  const std::size_t filter_offset = M * K;
  // number of values in an output region / column
  const std::size_t top_offset = M * N;

  const int* blur_neighbors = 0;
  if (size > 1) blur_neighbors = blur_neighbors_.gpu_data();

  value_type* const col_data      = col_data_blob.mutable_gpu_data();
  value_type* const col_diff_data = col_data_blob.mutable_gpu_diff();

  for (std::size_t g = 0; g < group_; ++g) {
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::size_t start = c * chunk_size;
      const std::size_t end   = std::min<std::size_t>(N, start + chunk_size);

      im2col(splatted_.gpu_data() + (value_size_ / group_) * g * (N + 1),
             value_size_ / group_, size, M_, start, end, blur_neighbors,
             col_data);

      // Gradient w.r.t. filter.
      caffe_gpu_gemm_ex<value_type>(
        CblasNoTrans, CblasTrans, M, K, end - start, static_cast<value_type>(1),
        blurred_tick.gpu_data() + top_offset * g + chunk_size * c, N, col_data,
        end - start, static_cast<value_type>(1), filter_out + filter_offset * g,
        K);

      // Gradient w.r.t. data.
      caffe_gpu_gemm_ex<value_type>(
        CblasTrans, CblasNoTrans, K, end - start, M, static_cast<value_type>(1),
        filter_.gpu_data() + filter_offset * g, K,
        blurred_tick.gpu_data() + top_offset * g + chunk_size * c, N,
        static_cast<value_type>(0), col_diff_data, end - start);

      col2im(
        col_diff_data, value_size_ / group_, size, M_, start, end,
        blur_neighbors,
        blurred_out->mutable_gpu_data() + (value_size_ / group_) * g * (N + 1));
    }
  }
}

template <typename T>
__global__ void maxtick_gpu_kernel(const T* tick_data,
                                   const int value_size,
                                   const int filter_size,
                                   const int M,
                                   const int* blur_neighbors,
                                   T* max_out_tick_data,
                                   const int* idx_data) {
  CUDA_KERNEL_LOOP(i, value_size) {
    // Looping over variables
    for (std::size_t j = 0; j < M; ++j) {
      // Looping only over the neighbors
      if (idx_data[i * M + j] == i * (M + 1) + j + 1) {
        max_out_tick_data[i * (M + 1) + j + 1] += tick_data[i * M + j];
      }
      for (std::size_t k = 1; k < filter_size; ++k) {
        const int* neighbors = blur_neighbors + ((k - 1) * M);
        if (idx_data[i * M + j] == i * (M + 1) + neighbors[j] + 1) {
          max_out_tick_data[i * (M + 1) + neighbors[j] + 1] +=
            tick_data[i * M + j];
        }
      }
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::max_tick(const Blob<value_type>& maxxed_tick,
                                          Blob<value_type>* maxxed_out) {
  // filter_       num_output x value_size / group x filter_size row-major
  // maxxed_out    value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // maxxed_tick   value_size x M_                               row-major

  maxxed_out->Reshape(1, 1, value_size_, M_ + 1);
  caffe_gpu_memset(value_size_ * (M_ + 1) * sizeof(value_type), 0,
                   maxxed_out->mutable_gpu_data());

  const value_type* tick_data = maxxed_tick.gpu_data();

  const int filter_size = get_filter_size(neighborhood_size_, d_);

  const int* blur_neighbors = 0;
  if (filter_size > 1) blur_neighbors = blur_neighbors_.gpu_data();

  value_type* max_out_tick_data = maxxed_out->mutable_gpu_data();

  maxtick_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(value_size_), CAFFE_CUDA_NUM_THREADS>>>(
      tick_data, value_size_, filter_size, M_, blur_neighbors,
      max_out_tick_data, max_idx_.gpu_data());

  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void splat_tick_gpu_kernel(const int in_size,
                                      const T* splatted,
                                      const int M,
                                      const int d,
                                      const int in_offset,
                                      const int value_size,
                                      const int* offset,
                                      const T* barycentric,
                                      T* in) {
  typedef T value_type;

  CUDA_KERNEL_LOOP(i, in_size) {
    for (int j = 0; j <= d; ++j) {
      int o               = offset[(in_offset + i) * (d + 1) + j] + 1;
      const value_type& w = barycentric[(in_offset + i) * (d + 1) + j];

      for (int k = 0; k < value_size; k++) {
        in[k * in_size + i] += w * splatted[k * (M + 1) + o];
      }
    }
  }
}

template <typename T>
void PermutohedralReverseGpu<T>::splat_tick(
  const Blob<value_type>& splatted_tick, value_type* splatted_out) {
  // splatted_tick  value_size x (M_+1)                           row-major
  // splatted_out   value_size x in_size                          row-major
  splat_tick_gpu_kernel<value_type>
    // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(in_size_), CAFFE_CUDA_NUM_THREADS>>>(
      in_size_, splatted_tick.gpu_data(), M_, d_, in_offset_, value_size_,
      offset_.gpu_data(), barycentric_.gpu_data(), splatted_out);
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
boost::shared_ptr<PermutohedralReverse<T>> Permutohedral<T>::compute_gpu(
  const value_type* filter,
  const value_type* in,
  int num_output,
  int group,
  int value_size,
  bool do_skip_blur,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  value_type* out) const {
  // Setup blur operation. This op will be returned to be able to compute the
  // gradient later.
  // TODO(mkiefel): probably move to some kind of constructor or init.
  boost::shared_ptr<PermutohedralReverseGpu<value_type>> reverse_operation(
    new PermutohedralReverseGpu<value_type>());

  reverse_operation->init(filter, num_output, group, value_size, do_skip_blur,
                          in_offset, out_offset, in_size, out_size, lattice_);
  reverse_operation->compute(in, out);

  return reverse_operation;
}

template <typename T>
boost::shared_ptr<PermutohedralReverse<T>> Permutohedral<T>::max_compute_gpu(
  const value_type* filter,
  const value_type* in,
  int value_size,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  value_type* out) {
  // Setup max operation. This op will be returned to be able to compute the
  // gradient later.
  // TODO(mkiefel): probably move to some kind of constructor or init.
  boost::shared_ptr<PermutohedralReverseGpu<value_type>> reverse_operation(
    new PermutohedralReverseGpu<value_type>());

  reverse_operation->init(filter, value_size, value_size, value_size, false,
                          in_offset, out_offset, in_size, out_size, lattice_);

  reverse_operation->max_compute(in, out);

  return reverse_operation;
}

// explicit instantiation
template boost::shared_ptr<PermutohedralReverse<double>>
Permutohedral<double>::compute_gpu(const double* filter,
                                   const double* in,
                                   int num_input,
                                   int group,
                                   int value_size,
                                   bool do_skip_blur,
                                   int in_offset,
                                   int out_offset,
                                   int in_size,
                                   int out_size,
                                   double* out) const;
template boost::shared_ptr<PermutohedralReverse<float>>
Permutohedral<float>::compute_gpu(const float* filter,
                                  const float* in,
                                  int num_input,
                                  int group,
                                  int value_size,
                                  bool do_skip_blur,
                                  int in_offset,
                                  int out_offset,
                                  int in_size,
                                  int out_size,
                                  float* out) const;

template boost::shared_ptr<PermutohedralReverse<double>>
Permutohedral<double>::max_compute_gpu(const double* filter,
                                       const double* in,
                                       int value_size,
                                       int in_offset,
                                       int out_offset,
                                       int in_size,
                                       int out_size,
                                       double* out);
template boost::shared_ptr<PermutohedralReverse<float>>
Permutohedral<float>::max_compute_gpu(const float* filter,
                                      const float* in,
                                      int value_size,
                                      int in_offset,
                                      int out_offset,
                                      int in_size,
                                      int out_size,
                                      float* out);

}  // namespace permutohedral

}  // namespace caffe
