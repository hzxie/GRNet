// Copyright 2019 Haozhe Xie
// Distributed under the MIT Software license,
// (See https://opensource.org/licenses/MIT)

#ifndef CAFFE_CHAMFER_DISTANCE_LAYER_HPP_
#define CAFFE_CHAMFER_DISTANCE_LAYER_HPP_

#include <iostream>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class ChamferDistanceLossLayer : public LossLayer<Dtype> {
 public:
  explicit ChamferDistanceLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ChamferDistanceLoss"; }
  /**
   * Unlike most loss layers, in the ChamferDistanceLossLayer we can
   * backpropagate to both inputs -- override to return true and always allow
   * force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> dist1_;
  Blob<Dtype> dist2_;
  Blob<int> indexes1_;
  Blob<int> indexes2_;
};

}  // namespace caffe

#endif  // CAFFE_CHAMFER_DISTANCE_LAYER_HPP_
