// Copyright 2015 MPI Tuebingen

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "boost/array.hpp"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/permutohedral_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PermutohedralPoolingLayerTest : public MultiDeviceTest<TypeParam> {
 public:
  typedef typename TypeParam::Dtype Dtype;
  PermutohedralPoolingLayerTest()
    : data_(new Blob<Dtype>(1, 2, 5, 6)),
      features_(new Blob<Dtype>()),
      color_features_(new Blob<Dtype>()),
      big_data_(new Blob<Dtype>(1, 2, 50, 60)),
      big_features_(new Blob<Dtype>()),
      big_color_features_(new Blob<Dtype>()),
      blurred_(new Blob<Dtype>()) {
    Caffe::set_random_seed(2305);

    Fill(data_, features_, color_features_, &bottom_, &color_bottom_);
    Fill(big_data_, big_features_, big_color_features_, &big_bottom_,
         &big_color_bottom_);

    top_.push_back(blurred_.get());
  }

 protected:
  static void Fill(const shared_ptr<Blob<Dtype> >& data,
                   const shared_ptr<Blob<Dtype> >& features,
                   const shared_ptr<Blob<Dtype> >& color_features,
                   std::vector<Blob<Dtype>*>* bottom,
                   std::vector<Blob<Dtype>*>* color_bottom) {
    const Dtype sigma       = 5;
    const Dtype color_sigma = 0.3;

    const int num      = data->num();
    const int channels = data->channels();
    const int height   = data->height();
    const int width    = data->width();

    features->Reshape(num, 2, height, width);
    Dtype* features_data = features->mutable_cpu_data();

    color_features->Reshape(num, 3, height, width);
    Dtype* color_features_data = color_features->mutable_cpu_data();

    for (int n = 0; n < num; ++n) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          features_data[features->offset(n, 0, y, x)] = y / sigma;
          features_data[features->offset(n, 1, y, x)] = x / sigma;

          color_features_data[color_features->offset(n, 0, y, x)] = y / sigma;
          color_features_data[color_features->offset(n, 1, y, x)] = x / sigma;
          color_features_data[color_features->offset(n, 2, y, x)] =
            (caffe_rng_rand() % 100) / 100.0 / color_sigma;
        }
      }
    }

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(data.get());
    Dtype* data_data = data->mutable_cpu_data();
    // Build a safe margin around the data. This is especially necessary for the
    // GaussPermutohedral implementation: It works better if more cells around
    // the true data are allocated.
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < 0.2 * height; ++y) {
          for (int x = 0; x < 0.2 * width; ++x) {
            data_data[((n * channels + c) * height + y) * width + x] = 0;
          }
          for (int x = 0.8 * width; x < width; ++x) {
            data_data[((n * channels + c) * height + y) * width + x] = 0;
          }
        }

        for (int y = 0.8 * height; y < height; ++y) {
          for (int x = 0; x < 0.2 * width; ++x) {
            data_data[((n * channels + c) * height + y) * width + x] = 0;
          }
          for (int x = 0.8 * width; x < width; ++x) {
            data_data[((n * channels + c) * height + y) * width + x] = 0;
          }
        }
      }
    }

    bottom->push_back(data.get());
    bottom->push_back(features.get());
    bottom->push_back(features.get());

    color_bottom->push_back(data.get());
    color_bottom->push_back(color_features.get());
    color_bottom->push_back(color_features.get());
  }

  shared_ptr<Blob<Dtype> > data_;
  shared_ptr<Blob<Dtype> > features_;
  shared_ptr<Blob<Dtype> > color_features_;

  shared_ptr<Blob<Dtype> > big_data_;
  shared_ptr<Blob<Dtype> > big_features_;
  shared_ptr<Blob<Dtype> > big_color_features_;

  shared_ptr<Blob<Dtype> > blurred_;

  std::vector<Blob<Dtype>*> bottom_;
  std::vector<Blob<Dtype>*> color_bottom_;

  std::vector<Blob<Dtype>*> big_bottom_;
  std::vector<Blob<Dtype>*> big_color_bottom_;

  std::vector<Blob<Dtype>*> top_;
};

template <typename Dtype, template <typename> class Ptype>
struct TestSuite {
  static void TestSetup(const std::vector<Blob<Dtype>*>& bottom,
                        const std::vector<Blob<Dtype>*>& top,
                        int neighborhood_size,
                        int num_output,
                        int group) {
    LayerParameter layer_param;
    PermutohedralPoolingParameter* permutohedral_pooling_param =
      layer_param.mutable_permutohedral_pooling_param();
    permutohedral_pooling_param->set_neighborhood_size(neighborhood_size);

    permutohedral_pooling_param->set_repeated_init(false);

    PermutohedralPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom, top);
  }

  static void TestGradient(const std::vector<Blob<Dtype>*>& bottom,
                           const std::vector<Blob<Dtype>*>& top,
                           int neighborhood_size,
                           int num_output,
                           int group) {
    LayerParameter layer_param;
    PermutohedralPoolingParameter* permutohedral_pooling_param =
      layer_param.mutable_permutohedral_pooling_param();
    // permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_pooling_param->set_neighborhood_size(neighborhood_size);
    // Remark, it's important to set this to false! CheckGradientExhaustive
    // wiggles also the features and this leads to a gradient with respect to
    // the output. This is not what we want. We cannot get the gradient for the
    // features, yet.
    permutohedral_pooling_param->set_repeated_init(false);

    PermutohedralPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom, top);

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    checker.CheckGradientExhaustive(&layer, bottom, top, 0);
  }
};

TYPED_TEST_CASE(PermutohedralPoolingLayerTest, TestDtypesAndDevices);

#define DEFINE_TYPED_TEST(test, type, neighborhood_size, num_output, group,    \
                          bottom)                                              \
  TYPED_TEST(                                                                  \
    PermutohedralPoolingLayerTest,                                             \
    test##_##type##_##neighborhood_size##_##num_output##_##group##_##bottom) { \
    typedef typename TypeParam::Dtype Dtype;                                   \
    TestSuite<Dtype, permutohedral::type>::test(                               \
      this->bottom, this->top_, neighborhood_size, num_output, group);         \
  }

DEFINE_TYPED_TEST(TestSetup, Permutohedral, 3, 1, 1, bottom_)
DEFINE_TYPED_TEST(TestGradient, Permutohedral, 3, 1, 1, bottom_)
// DEFINE_TYPED_TEST(TestGradient, Permutohedral,
//                   3, 2, 1, bottom_)
// DEFINE_TYPED_TEST(TestGradient, Permutohedral,
//                   3, 2, 2, bottom_)

}  // namespace caffe
