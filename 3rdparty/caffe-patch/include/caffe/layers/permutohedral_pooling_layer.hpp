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
template <typename Dtype>
class PermutohedralPoolingLayer : public Layer<Dtype> {
 public:
  explicit PermutohedralPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PermutohedralPooling"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  // virtual inline bool EqualNumBottomTopBlobs() const { return true; }

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
  bool do_repeated_init_;

  typedef permutohedral::Permutohedral<Dtype> permutohedral_type;

  struct MaxOperation {
    boost::shared_ptr<permutohedral_type> max_;
    boost::shared_ptr<permutohedral::PermutohedralReverse<Dtype> > reverse_;
    boost::shared_ptr<Blob<Dtype> > norm_there_;
    boost::shared_ptr<Blob<Dtype> > norm_back_;
  };

  std::vector<MaxOperation> operations_;
};

}  // namespace caffe

#endif  // PERMUTOHEDRAL_LAYERS_HPP_
