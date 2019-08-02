// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#ifndef PERMUTOHEDRAL_LAYERS_HPP_
#define PERMUTOHEDRAL_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/gauss_permutohedral.hpp"
#include "caffe/util/permutohedral.hpp"

namespace caffe {

/**
 * Permutohedral and GaussPermutohedral layers
 *
 */
template <template <typename> class Ptype>
struct PermutohedralTypeTraits;

template <>
struct PermutohedralTypeTraits<permutohedral::Permutohedral> {
  static const char* type() { return "Permutohedral"; }
};

template <>
struct PermutohedralTypeTraits<permutohedral::GaussPermutohedral> {
  static const char* type() { return "GaussPermutohedral"; }
};

template <typename Dtype, template <typename> class Ptype>
class PermutohedralLayerTemplate : public Layer<Dtype> {
 public:
  explicit PermutohedralLayerTemplate(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return PermutohedralTypeTraits<Ptype>::type();
  }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  void OffsetFilter(const Blob<Dtype>& filter_blob,
                    Blob<Dtype>* shifted_filter_blob) const;

  void InitLattice(const Blob<Dtype>* const in_lattice_blob,
                   const Blob<Dtype>& in_feature_blob,
                   const Blob<Dtype>& out_feature_blob,
                   Blob<Dtype>* const lattice_blob);

  int num_;
  int channels_;
  int in_height_, in_width_;
  int out_height_, out_width_;
  int feature_size_;

  int neighborhood_size_;
  int num_output_;
  int group_;

  bool do_repeated_init_;

  Blob<Dtype> bias_multiplier_;
  bool bias_term_;

  bool do_visualization_;

  bool do_skip_blur_;

  typedef Ptype<Dtype> permutohedral_type;

  struct BlurOperation {
    boost::shared_ptr<permutohedral_type> blur_;
    boost::shared_ptr<permutohedral::PermutohedralReverse<Dtype> > reverse_;
    boost::shared_ptr<Blob<Dtype> > norm_there_;
    boost::shared_ptr<Blob<Dtype> > norm_back_;
  };

  std::vector<BlurOperation> operations_;
};

}  // namespace caffe

#endif  // PERMUTOHEDRAL_LAYERS_HPP_
