// Copyright 2019 Haozhe Xie
// Distributed under the MIT Software license,
// (See https://opensource.org/licenses/MIT)

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/chamfer_distance_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ChamferDistanceLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ChamferDistanceLossLayerTest()
    : blob_bottom_ptcloud_(new Blob<Dtype>(2, 3, 3, 1)),
      blob_bottom_gtcloud_(new Blob<Dtype>(2, 5, 3, 1)),
      blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);

    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_ptcloud_);
    filler.Fill(this->blob_bottom_gtcloud_);

    blob_bottom_vec_.push_back(blob_bottom_ptcloud_);
    blob_bottom_vec_.push_back(blob_bottom_gtcloud_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~ChamferDistanceLossLayerTest() {
    delete blob_bottom_ptcloud_;
    delete blob_bottom_gtcloud_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_ptcloud_;
  Blob<Dtype>* const blob_bottom_gtcloud_;
  Blob<Dtype>* const blob_top_loss_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ChamferDistanceLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ChamferDistanceLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ChamferDistanceLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype cd_loss =
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Dtype loss(0);
  const int num       = this->blob_bottom_gtcloud_->num();
  const int n_points1 = this->blob_bottom_ptcloud_->channels();
  const int n_points2 = this->blob_bottom_gtcloud_->channels();

  for (int i = 0; i < num; ++i) {
    Dtype min_distance1(0);
    Dtype min_distance2(0);

    for (int j = 0; j < n_points1; ++j) {
      Dtype _min_distance(1e12);
      for (int k = 0; k < n_points2; ++k) {
        Dtype x1   = this->blob_bottom_ptcloud_->data_at(i, j, 0, 0);
        Dtype y1   = this->blob_bottom_ptcloud_->data_at(i, j, 1, 0);
        Dtype z1   = this->blob_bottom_ptcloud_->data_at(i, j, 2, 0);
        Dtype x2   = this->blob_bottom_gtcloud_->data_at(i, k, 0, 0);
        Dtype y2   = this->blob_bottom_gtcloud_->data_at(i, k, 1, 0);
        Dtype z2   = this->blob_bottom_gtcloud_->data_at(i, k, 2, 0);
        Dtype dx   = x1 - x2;
        Dtype dy   = y1 - y2;
        Dtype dz   = z1 - z2;
        Dtype dist = dx * dx + dy * dy + dz * dz;
        if (dist < _min_distance) {
          _min_distance = dist;
        }
      }
      min_distance1 += _min_distance;
    }

    for (int j = 0; j < n_points2; ++j) {
      Dtype _min_distance(1e12);
      for (int k = 0; k < n_points1; ++k) {
        Dtype x1   = this->blob_bottom_gtcloud_->data_at(i, j, 0, 0);
        Dtype y1   = this->blob_bottom_gtcloud_->data_at(i, j, 1, 0);
        Dtype z1   = this->blob_bottom_gtcloud_->data_at(i, j, 2, 0);
        Dtype x2   = this->blob_bottom_ptcloud_->data_at(i, k, 0, 0);
        Dtype y2   = this->blob_bottom_ptcloud_->data_at(i, k, 1, 0);
        Dtype z2   = this->blob_bottom_ptcloud_->data_at(i, k, 2, 0);
        Dtype dx   = x1 - x2;
        Dtype dy   = y1 - y2;
        Dtype dz   = z1 - z2;
        Dtype dist = dx * dx + dy * dy + dz * dz;
        if (dist < _min_distance) {
          _min_distance = dist;
        }
      }
      min_distance2 += _min_distance;
    }
    min_distance1 /= n_points1;
    min_distance2 /= n_points2;
    loss += min_distance1 + min_distance2;
  }
  loss /= num;
  EXPECT_NEAR(cd_loss, loss, 1e-5);
}

TYPED_TEST(ChamferDistanceLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ChamferDistanceLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}  // namespace caffe
