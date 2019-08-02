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

#ifndef CAFFE_UTIL_GAUSS_PERMUTOHEDRAL_H_
#define CAFFE_UTIL_GAUSS_PERMUTOHEDRAL_H_

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/array.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/util/permutohedral.hpp"

namespace caffe {
namespace permutohedral {

/************************************************/
/***         Constant Gauss Filter            ***/
/************************************************/

/*! \brief Class for constant Gauss filter.
 */

template <typename TValue>
class ConstFilter {
 public:
  typedef TValue value_type;

  ConstFilter(int /*neighbourhood_size*/, int feature_size)
    : weight_(1.0 / (1 + std::pow(2.0, -feature_size))) {}

  const value_type* filter() const { return &weight_; }

 private:
  value_type weight_;
};

/************************************************/
/***    Gauss Permutohedral Lattice           ***/
/************************************************/

/*! \brief This is the main class for lattice construction and forward/backward
 * operations in 'separable' Gaussian sparse high dimensional filtering.
 *
 *  Since the filter is fixed to be Gaussian, there is no filter learning, but
 *  the loss is back-propabale with 'reverse' functions.
 *  At present, there is no dedicated GPU computations for this layer.
 *  Below GPU functions does the computations in CPU by copying memory to/from
 * CPU.
 *
 *  Parts of the code are adapted and modified from the separable filter code
 * from Adams et al. 2010 (http://graphics.stanford.edu/papers/permutohedral/).
 */

template <typename TValue>
class GaussPermutohedral {
 public:
  typedef TValue value_type;
  typedef ConstFilter<value_type> gauss_type;

 private:
  struct Neighbors {
    int n1, n2;
  };

  // Lattice structure
  struct Lattice {
    std::vector<value_type> barycentric_;
    std::vector<int> offset_;
    std::vector<Neighbors> blur_neighbors_;
    Lattice(std::size_t s) {
      offset_.resize(s);
      barycentric_.resize(s);
    }
  };

  boost::shared_ptr<const Lattice> lattice_;

  // Number of elements, size of sparse discretized space, dimension of features
  int N_, M_, d_;

 public:
  class GaussPermutohedralReverse : public PermutohedralReverse<value_type> {
   public:
    void reverse(const value_type* diff_in,
                 value_type* diff_out_filter,
                 value_type* diff_out_in);

    void max_reverse(const value_type* diff_in, value_type* diff_out_in){};

   private:
    // don't copy
    GaussPermutohedralReverse(const GaussPermutohedralReverse& rhs);

    GaussPermutohedralReverse() {}

    void seqCompute(value_type* out,
                    const value_type* in,
                    int in_offset,
                    int out_offset,
                    int in_size,
                    int out_size,
                    int value_size,
                    bool reverse = false) const;

    std::vector<value_type> value_;

    int d_, N_;
    int M_;
    boost::shared_ptr<const Lattice> lattice_;

    int in_offset_, out_offset_, in_size_, out_size_;
    int value_size_;

    std::vector<value_type> filter_;

    friend class GaussPermutohedral;
  };

#ifndef CPU_ONLY
  class GaussPermutohedralReverseGpu : public PermutohedralReverse<value_type> {
   public:
    void reverse(const value_type* diff_in,
                 value_type* diff_out_filter,
                 value_type* diff_out_in);

    void max_reverse(const value_type* diff_in, value_type* diff_out_in){};

   private:
    GaussPermutohedralReverseGpu(
      const boost::shared_ptr<PermutohedralReverse<value_type> >& op,
      const int in_size,
      const int out_size,
      const int value_size)
      : op_(op),
        in_size_(in_size),
        out_size_(out_size),
        value_size_(value_size) {}

    boost::shared_ptr<PermutohedralReverse<value_type> > op_;
    int in_size_, out_size_;
    int value_size_;

    friend class GaussPermutohedral;
  };
#endif

  GaussPermutohedral();

  static int get_filter_size(int neighborhood_size, int feature_size);

  void init(const value_type* feature,
            int data_count,
            int feature_size,
            int neighborhood_size,
            bool do_visualization);

  boost::shared_ptr<PermutohedralReverse<value_type> > compute(
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
    value_type* out) const;

  boost::shared_ptr<PermutohedralReverse<value_type> > compute_gpu(
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
    value_type* out) const;
};

}  // namespace permutohedral

}  // namespace caffe

#endif /* CAFFE_UTIL_GAUSS_PERMUTOHEDRAL_H */
