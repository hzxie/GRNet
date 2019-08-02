// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pixel_feature_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class PixelFeatureLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PixelFeatureLayerTest()
    : blob_bottom_0_(new Blob<Dtype>(1, 2, 4, 3)),
      blob_bottom_0_2_(new Blob<Dtype>(1, 1, 3, 3)),
      blob_bottom_0_3_(new Blob<Dtype>(2, 2, 4, 3)),
      blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_2_.push_back(blob_bottom_0_2_);
    blob_bottom_vec_3_.push_back(blob_bottom_0_3_);
    blob_top_vec_.push_back(blob_top_);

    for (int i = 0; i < blob_bottom_0_->count(); i++) {
      blob_bottom_0_->mutable_cpu_data()[i] = i * 10;
    }

    for (int i = 0; i < blob_bottom_0_2_->count(); i++) {
      blob_bottom_0_2_->mutable_cpu_data()[i] = i + 3;
    }

    for (int i = 0; i < blob_bottom_0_3_->count(); i++) {
      blob_bottom_0_3_->mutable_cpu_data()[i] = (i % 24) * 10;
    }
  }
  virtual ~PixelFeatureLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_0_2_;
    delete blob_bottom_0_3_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_0_2_;
  Blob<Dtype>* const blob_bottom_0_3_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_3_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PixelFeatureLayerTest, TestDtypesAndDevices);

TYPED_TEST(PixelFeatureLayerTest, TestSetUp1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_POSITION_AND_RGB);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(PixelFeatureLayerTest, TestSetUp2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_POSITION);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(PixelFeatureLayerTest, TestSetUp3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_RGB);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

//
TYPED_TEST(PixelFeatureLayerTest, TestForwardXY) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_POSITION);
  pixel_param->set_pos_scale(2.5);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_3_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[1], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[2], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[3], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[4], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[5], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[6], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[7], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[8], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[9], 3 * 2.5, 1e-4);
  EXPECT_NEAR(data[10], 3 * 2.5, 1e-4);
  EXPECT_NEAR(data[11], 3 * 2.5, 1e-4);

  EXPECT_NEAR(data[12], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[13], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[14], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[15], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[16], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[17], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[18], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[19], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[20], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[21], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[22], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[23], 2 * 2.5, 1e-4);

  EXPECT_NEAR(data[24], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[25], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[26], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[27], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[28], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[29], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[30], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[31], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[32], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[33], 3 * 2.5, 1e-4);
  EXPECT_NEAR(data[34], 3 * 2.5, 1e-4);
  EXPECT_NEAR(data[35], 3 * 2.5, 1e-4);

  EXPECT_NEAR(data[36], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[37], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[38], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[39], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[40], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[41], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[42], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[43], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[44], 2 * 2.5, 1e-4);
  EXPECT_NEAR(data[45], 0 * 2.5, 1e-4);
  EXPECT_NEAR(data[46], 1 * 2.5, 1e-4);
  EXPECT_NEAR(data[47], 2 * 2.5, 1e-4);
}

TYPED_TEST(PixelFeatureLayerTest, TestForwardXY2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_POSITION);
  pixel_param->set_pos_scale(-3.4);
  pixel_param->set_offset_h(2.5);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[1], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[2], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[3], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[4], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[5], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[6], 2 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[7], 2 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[8], 2 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[9], 3 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[10], 3 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[11], 3 * -3.4 + 2.5, 1e-4);

  EXPECT_NEAR(data[12], 0 * -3.4, 1e-4);
  EXPECT_NEAR(data[13], 1 * -3.4, 1e-4);
  EXPECT_NEAR(data[14], 2 * -3.4, 1e-4);
  EXPECT_NEAR(data[15], 0 * -3.4, 1e-4);
  EXPECT_NEAR(data[16], 1 * -3.4, 1e-4);
  EXPECT_NEAR(data[17], 2 * -3.4, 1e-4);
  EXPECT_NEAR(data[18], 0 * -3.4, 1e-4);
  EXPECT_NEAR(data[19], 1 * -3.4, 1e-4);
  EXPECT_NEAR(data[20], 2 * -3.4, 1e-4);
  EXPECT_NEAR(data[21], 0 * -3.4, 1e-4);
  EXPECT_NEAR(data[22], 1 * -3.4, 1e-4);
  EXPECT_NEAR(data[23], 2 * -3.4, 1e-4);
}

TYPED_TEST(PixelFeatureLayerTest, TestForwardRGB1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_RGB);
  pixel_param->set_color_scale(-3.4);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  for (int i = 0; i < 24; ++i) {
    EXPECT_NEAR(data[i], i * 10 * -3.4, 1e-4);
  }
}

TYPED_TEST(PixelFeatureLayerTest, TestForwardXYRGB1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_POSITION_AND_RGB);
  pixel_param->set_pos_scale(-3.4);
  pixel_param->set_offset_w(2.5);
  pixel_param->set_offset_h(1.5);
  pixel_param->set_color_scale(-3.4);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_2_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0], 0 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[1], 0 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[2], 0 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[3], 1 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[4], 1 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[5], 1 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[6], 2 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[7], 2 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[8], 2 * -3.4 + 1.5, 1e-4);

  EXPECT_NEAR(data[9], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[10], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[11], 2 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[12], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[13], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[14], 2 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[15], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[16], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[17], 2 * -3.4 + 2.5, 1e-4);
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(data[i + 18], (i + 3) * -3.4, 1e-4);
  }
}

TYPED_TEST(PixelFeatureLayerTest, TestForwardRGBXY1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PixelFeatureParameter* pixel_param =
    layer_param.mutable_pixel_feature_param();
  pixel_param->set_type(PixelFeatureParameter_Feature_RGB_AND_POSITION);
  pixel_param->set_pos_scale(-3.4);
  pixel_param->set_offset_w(2.5);
  pixel_param->set_offset_h(1.5);
  pixel_param->set_color_scale(-3.4);
  shared_ptr<PixelFeatureLayer<Dtype> > layer(
    new PixelFeatureLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_2_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(data[i], (i + 3) * -3.4, 1e-4);
  }

  // for every top blob element ...
  EXPECT_NEAR(data[0 + 9], 0 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[1 + 9], 0 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[2 + 9], 0 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[3 + 9], 1 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[4 + 9], 1 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[5 + 9], 1 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[6 + 9], 2 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[7 + 9], 2 * -3.4 + 1.5, 1e-4);
  EXPECT_NEAR(data[8 + 9], 2 * -3.4 + 1.5, 1e-4);

  EXPECT_NEAR(data[9 + 9], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[10 + 9], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[11 + 9], 2 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[12 + 9], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[13 + 9], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[14 + 9], 2 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[15 + 9], 0 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[16 + 9], 1 * -3.4 + 2.5, 1e-4);
  EXPECT_NEAR(data[17 + 9], 2 * -3.4 + 2.5, 1e-4);
}

}  // namespace caffe
