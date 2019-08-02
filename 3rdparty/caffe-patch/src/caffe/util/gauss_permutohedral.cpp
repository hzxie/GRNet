// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
   THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "caffe/util/gauss_permutohedral.hpp"

#include <algorithm>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/permutohedral.hpp"

#include "boost/make_shared.hpp"

namespace caffe {
namespace permutohedral {

/************************************************/
/***    Gauss Permutohedral Lattice           ***/
/************************************************/

template <typename TValue>
GaussPermutohedral<TValue>::GaussPermutohedral() : N_(0), M_(0), d_(0) {}

// this part contains code from Philipp Krähenbühl
template <typename TValue>
void GaussPermutohedral<TValue>::init(const value_type* feature,
                                      int data_count,
                                      int feature_size,
                                      int neighborhood_size,
                                      bool do_visualization) {
  CHECK_EQ(neighborhood_size, 2) << "neighborhood_size has to be 2";

  // Compute the lattice coordinates for each feature [there is going to be a
  // lot of magic here
  N_ = data_count;
  d_ = feature_size;
  HashTable hash_table(d_, N_ * (d_ + 1));

  boost::shared_ptr<Lattice> lattice =
    boost::make_shared<Lattice>((d_ + 1) * N_);
  lattice_ = lattice;

  // Allocate the local memory
  std::vector<value_type> scale_factor(d_);
  std::vector<value_type> elevated(d_ + 1);
  std::vector<value_type> rem0(d_ + 1);
  std::vector<value_type> barycentric(d_ + 2);
  std::vector<short> rank(d_ + 1);
  std::vector<short> canonical((d_ + 1) * (d_ + 1));
  std::vector<short> key(d_ + 1);

  // Compute the canonical simplex
  for (int i = 0; i <= d_; i++) {
    for (int j = 0; j <= d_ - i; j++) canonical[i * (d_ + 1) + j] = i;
    for (int j = d_ - i + 1; j <= d_; j++)
      canonical[i * (d_ + 1) + j] = i - (d_ + 1);
  }

  // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
  value_type inv_std_dev =
    std::sqrt(static_cast<value_type>(2.0 / 3.0)) * (d_ + 1);
  // Compute the diagonal part of E (p.5 in [Adams etal 2010])
  for (int i = 0; i < d_; i++)
    scale_factor[i] = static_cast<value_type>(1.0) /
                      sqrt(static_cast<value_type>((i + 2) * (i + 1))) *
                      inv_std_dev;

  const value_type* f = feature;

  // Compute the simplex each feature lies in
  for (int k = 0; k < N_; k++) {
    // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
    // sm contains the sum of 1..n of our faeture vector
    value_type sm = 0;
    for (int j = d_; j > 0; j--) {
      const int fIndex = (j - 1) * N_ + k;
      value_type cf    = f[fIndex] * scale_factor[j - 1];
      elevated[j]      = sm - j * cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    value_type down_factor = static_cast<value_type>(1.0) / (d_ + 1);
    value_type up_factor   = (d_ + 1);
    int sum                = 0;
    for (int i = 0; i <= d_; i++) {
      int rd2;
      value_type v    = down_factor * elevated[i];
      value_type up   = std::ceil(v) * up_factor;
      value_type down = std::floor(v) * up_factor;
      if (up - elevated[i] < elevated[i] - down)
        rd2 = (short)up;
      else
        rd2 = (short)down;

      rem0[i] = rd2;
      sum += rd2 * down_factor;
    }

    // Find the simplex we are in and store it in rank (where rank describes
    // what position coorinate i has in the sorted order of the features values)
    for (int i = 0; i <= d_; i++) rank[i] = 0;
    for (int i = 0; i < d_; i++) {
      value_type di = elevated[i] - rem0[i];
      for (int j = i + 1; j <= d_; j++)
        if (di < elevated[j] - rem0[j])
          rank[i]++;
        else
          rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= d_; i++) {
      rank[i] += sum;
      if (rank[i] < 0) {
        rank[i] += d_ + 1;
        rem0[i] += d_ + 1;
      } else if (rank[i] > d_) {
        rank[i] -= d_ + 1;
        rem0[i] -= d_ + 1;
      }
    }

    // If do_visualization is true, fill barycentric weights with 1.0
    // Otherwise, comptue the barycentric coordinates (p.10 in [Adams et al.
    // 2010])
    if (do_visualization) {
      for (int i = 0; i <= d_ + 1; i++) {
        barycentric[i] = 1.0;
      }
    } else {
      for (int i = 0; i <= d_ + 1; i++) barycentric[i] = 0;
      for (int i = 0; i <= d_; i++) {
        value_type v = (elevated[i] - rem0[i]) * down_factor;
        if (d_ - rank[i] < 0 || d_ - rank[i] + 1 >= d_ + 2)
          throw std::runtime_error("GaussPermutohedral: rank access error");
        barycentric[d_ - rank[i]] += v;
        barycentric[d_ - rank[i] + 1] -= v;
      }
      // Wrap around
      barycentric[0] += 1.0 + barycentric[d_ + 1];
    }

    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= d_ + 1; i++) barycentric[i] = 0;
    for (int i = 0; i <= d_; i++) {
      value_type v = (elevated[i] - rem0[i]) * down_factor;
      if (d_ - rank[i] < 0 || d_ - rank[i] + 1 >= d_ + 2)
        throw std::runtime_error("GaussPermutohedral: rank access error");
      barycentric[d_ - rank[i]] += v;
      barycentric[d_ - rank[i] + 1] -= v;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[d_ + 1];

    // Compute all vertices and their offset
    for (int remainder = 0; remainder <= d_; remainder++) {
      for (int i = 0; i < d_; i++)
        key[i] = rem0[i] + canonical[remainder * (d_ + 1) + rank[i]];
      lattice->offset_[k * (d_ + 1) + remainder] =
        hash_table.find(key.data(), true);
      lattice->barycentric_[k * (d_ + 1) + remainder] = barycentric[remainder];
    }
  }

  // Find the Neighbors of each lattice points
  // Get the number of vertices in the lattice
  M_ = hash_table.size();

  // Create the neighborhood structure
  lattice->blur_neighbors_.resize((d_ + 1) * M_);

  std::vector<short> n1(d_ + 1);
  std::vector<short> n2(d_ + 1);

  // For each of d+1 axes,
  for (int j = 0; j <= d_; j++) {
    for (int i = 0; i < M_; i++) {
      const short* key = hash_table.getKey(i);
      for (int k = 0; k < d_; k++) {
        n1[k] = key[k] - 1;
        n2[k] = key[k] + 1;
      }
      n1[j] = key[j] + d_;
      n2[j] = key[j] - d_;

      lattice->blur_neighbors_[j * M_ + i].n1 = hash_table.find(n1.data());
      lattice->blur_neighbors_[j * M_ + i].n2 = hash_table.find(n2.data());
    }
  }
}

template <typename TValue>
void GaussPermutohedral<TValue>::GaussPermutohedralReverse::seqCompute(
  value_type* out,
  const value_type* in,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  int value_size,
  bool reverse) const {
  // Shift all values by 1 such that -1 -> 0 (used for blurring)
  value_type* values     = new value_type[(M_ + 2) * value_size];
  value_type* new_values = new value_type[(M_ + 2) * value_size];

  value_type* in_filter  = new value_type[filter_.size()];
  value_type* out_filter = new value_type[filter_.size()];

  if (reverse) {
    std::copy(filter_.begin(), filter_.end(), in_filter);
    std::fill(out_filter, out_filter + filter_.size(), 1);
  } else {
    std::copy(filter_.begin(), filter_.end(), out_filter);
    std::fill(in_filter, in_filter + filter_.size(), 1);
  }

  for (int i = 0; i < (M_ + 2) * value_size; i++) values[i] = new_values[i] = 0;

  // Splatting
  for (int i = 0; i < in_size; i++) {
    for (int j = 0; j <= d_; j++) {
      int o        = lattice_->offset_[(i + in_offset) * (d_ + 1) + j] + 1;
      value_type w = lattice_->barycentric_[(i + in_offset) * (d_ + 1) + j];
      for (int k = 0; k < value_size; k++)
        values[o * value_size + k] += w * in[k * in_size + i] * in_filter[k];
    }
  }

  for (int j = reverse ? d_ : 0; j <= d_ && j >= 0; reverse ? j-- : j++) {
    values[0] = 0;
    for (int i = 0; i < M_; i++) {
      value_type* old_val = values + (i + 1) * value_size;
      value_type* new_val = new_values + (i + 1) * value_size;

      int n1             = lattice_->blur_neighbors_[j * M_ + i].n1 + 1;
      int n2             = lattice_->blur_neighbors_[j * M_ + i].n2 + 1;
      value_type* n1_val = values + n1 * value_size;
      value_type* n2_val = values + n2 * value_size;
      for (int k = 0; k < value_size; k++)
        new_val[k] = old_val[k] + 0.5 * (n1_val[k] + n2_val[k]);
    }
    std::swap(values, new_values);
  }

  // Slicing
  for (int i = 0; i < out_size; i++) {
    for (int k = 0; k < value_size; k++) out[k * out_size + i] = 0;
    for (int j = 0; j <= d_; j++) {
      int o        = lattice_->offset_[(i + out_offset) * (d_ + 1) + j] + 1;
      value_type w = lattice_->barycentric_[(i + out_offset) * (d_ + 1) + j];
      for (int k = 0; k < value_size; k++)
        out[k * out_size + i] += w * values[o * value_size + k] * out_filter[k];
    }
  }

  delete[] values;
  delete[] new_values;
}

template <typename TValue>
void GaussPermutohedral<TValue>::GaussPermutohedralReverse::reverse(
  const value_type* diff_in,
  value_type* diff_out_filter,
  value_type* diff_out_in) {
  seqCompute(diff_out_in, diff_in, out_offset_, in_offset_, out_size_, in_size_,
             value_size_, true);

  for (int k = 0; k < value_size_; k++) {
    diff_out_filter[k] = 0;
  }

  for (int i = 0; i < out_size_; i++) {
    for (int k = 0; k < value_size_; k++) {
      diff_out_filter[k] +=
        value_[k * out_size_ + i] / filter_[k] * diff_in[k * out_size_ + i];
    }
  }
}

#ifndef CPU_ONLY
template <typename TValue>
void GaussPermutohedral<TValue>::GaussPermutohedralReverseGpu::reverse(
  const value_type* diff_in,
  value_type* diff_out_filter,
  value_type* diff_out_in) {
  std::vector<value_type> diff_in_cpu(out_size_ * value_size_);
  caffe_gpu_memcpy(diff_in_cpu.size() * sizeof(value_type), diff_in,
                   diff_in_cpu.data());

  std::vector<value_type> diff_out_filter_cpu(value_size_);
  caffe_gpu_memcpy(diff_out_filter_cpu.size() * sizeof(value_type),
                   diff_out_filter, diff_out_filter_cpu.data());

  std::vector<value_type> diff_out_in_cpu(in_size_ * value_size_);
  caffe_gpu_memcpy(diff_out_in_cpu.size() * sizeof(value_type), diff_out_in,
                   diff_out_in_cpu.data());

  op_->reverse(diff_in_cpu.data(), diff_out_filter_cpu.data(),
               diff_out_in_cpu.data());

  caffe_gpu_memcpy(diff_out_filter_cpu.size() * sizeof(value_type),
                   diff_out_filter_cpu.data(), diff_out_filter);
  caffe_gpu_memcpy(diff_out_in_cpu.size() * sizeof(value_type),
                   diff_out_in_cpu.data(), diff_out_in);
}

template <typename TValue>
boost::shared_ptr<PermutohedralReverse<TValue> >
GaussPermutohedral<TValue>::compute_gpu(const value_type* filter,
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
  CHECK_EQ(num_output, value_size)
    << "number of outputs must match the number of inputs.";
  CHECK_EQ(group, value_size)
    << "this implementation only support a group count of the number of "
       "intput dimensions.";
  CHECK_EQ(do_skip_blur, false)
    << "You can not skip blur operation in Gauss Permutohedral layer.";

  std::vector<value_type> in_cpu(in_size * value_size);
  caffe_gpu_memcpy(in_cpu.size() * sizeof(value_type), in, in_cpu.data());

  std::vector<value_type> filter_cpu(value_size);
  caffe_gpu_memcpy(filter_cpu.size() * sizeof(value_type), filter,
                   filter_cpu.data());

  std::vector<value_type> out_cpu(out_size * value_size);

  boost::shared_ptr<PermutohedralReverse<value_type> > op = compute(
    filter_cpu.data(), in_cpu.data(), num_output, group, value_size,
    do_skip_blur, in_offset, out_offset, in_size, out_size, out_cpu.data());

  caffe_gpu_memcpy(out_cpu.size() * sizeof(value_type), out_cpu.data(), out);

  return boost::shared_ptr<PermutohedralReverse<value_type> >(
    new GaussPermutohedralReverseGpu(op, in_size, out_size, value_size));
}
#endif

template <typename TValue>
boost::shared_ptr<PermutohedralReverse<TValue> >
GaussPermutohedral<TValue>::compute(const value_type* filter,
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
  CHECK_EQ(num_output, value_size)
    << "number of outputs must match the number of inputs.";
  CHECK_EQ(group, value_size)
    << "this implementation only support a group count of the number of "
       "intput dimensions.";
  CHECK_EQ(do_skip_blur, false)
    << "You can not skip blur operation in Gauss Permutohedral layer.";

  boost::shared_ptr<GaussPermutohedralReverse> reverse_operation(
    new GaussPermutohedralReverse());

  reverse_operation->in_offset_  = in_offset;
  reverse_operation->out_offset_ = out_offset;
  reverse_operation->in_size_    = in_size;
  reverse_operation->out_size_   = out_size;

  reverse_operation->filter_.resize(value_size);
  std::copy(filter, filter + reverse_operation->filter_.size(),
            reverse_operation->filter_.begin());

  reverse_operation->lattice_ = lattice_;
  reverse_operation->d_       = d_;
  reverse_operation->N_       = N_;
  reverse_operation->M_       = M_;

  reverse_operation->value_size_ = value_size;

  reverse_operation->seqCompute(out, in, in_offset, out_offset, in_size,
                                out_size, value_size, false);

  // Copy the result of the blur operation for the gradient of the filter.
  reverse_operation->value_.resize(out_size * value_size);
  std::copy(out, out + reverse_operation->value_.size(),
            reverse_operation->value_.begin());

  return reverse_operation;
}

template <typename T>
int GaussPermutohedral<T>::get_filter_size(int /*neighborhood_size*/,
                                           int /*feature_size*/) {
  return 1;
}

INSTANTIATE_CLASS(GaussPermutohedral);

}  // namespace permutohedral

}  // namespace caffe
