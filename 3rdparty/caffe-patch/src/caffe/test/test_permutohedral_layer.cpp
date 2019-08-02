// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

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
#include "caffe/layers/permutohedral_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PermutohedralLayerTest : public MultiDeviceTest<TypeParam> {
 public:
  typedef typename TypeParam::Dtype Dtype;
  PermutohedralLayerTest()
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
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);
  }

  static void TestForward(const std::vector<Blob<Dtype>*>& bottom,
                          const std::vector<Blob<Dtype>*>& top,
                          int neighborhood_size,
                          int num_output,
                          int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);
    layer.Forward(bottom, top);
    CheckBruteForce(num_output, group, bottom, top);
  }

  static void TestForwardDiagOffset(const std::vector<Blob<Dtype>*>& bottom,
                                    const std::vector<Blob<Dtype>*>& top,
                                    int neighborhood_size,
                                    int num_output,
                                    int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);
    permutohedral_param->set_offset_type(
      PermutohedralParameter_OffsetType_DIAG);

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);
    layer.Forward(bottom, top);
    CheckBruteForce(num_output, num_output, bottom, top);
  }

  static void TestGradient(const std::vector<Blob<Dtype>*>& bottom,
                           const std::vector<Blob<Dtype>*>& top,
                           int neighborhood_size,
                           int num_output,
                           int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);
    // Remark, it's important to set this to false! CheckGradientExhaustive
    // wiggles also the features and this leads to a gradient with respect to
    // the output. This is not what we want. We cannot get the gradient for the
    // features, yet.
    permutohedral_param->set_repeated_init(false);
    permutohedral_param->set_bias_term(false);

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    checker.CheckGradientExhaustive(&layer, bottom, top, 0);
  }

  static void TestGradientBias(const std::vector<Blob<Dtype>*>& bottom,
                               const std::vector<Blob<Dtype>*>& top,
                               int neighborhood_size,
                               int num_output,
                               int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);
    // Remark, it's important to set this to false! CheckGradientExhaustive
    // wiggles also the features and this leads to a gradient with respect to
    // the output. This is not what we want. We cannot get the gradient for the
    // features, yet.
    permutohedral_param->set_repeated_init(false);
    permutohedral_param->set_bias_term(true);
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    checker.CheckGradientExhaustive(&layer, bottom, top, 0);
  }

  static void TestGradientBiasGroup(const std::vector<Blob<Dtype>*>& bottom,
                                    const std::vector<Blob<Dtype>*>& top,
                                    int neighborhood_size,
                                    int num_output,
                                    int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);
    // Remark, it's important to set this to false! CheckGradientExhaustive
    // wiggles also the features and this leads to a gradient with respect to
    // the output. This is not what we want. We cannot get the gradient for the
    // features, yet.
    permutohedral_param->set_repeated_init(false);
    permutohedral_param->set_bias_term(true);
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    checker.CheckGradientExhaustive(&layer, bottom, top, 0);
  }

  static void TestGradientBiasColor(const std::vector<Blob<Dtype>*>& bottom,
                                    const std::vector<Blob<Dtype>*>& top,
                                    int neighborhood_size,
                                    int num_output,
                                    int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);
    // Remark, it's important to set this to false! CheckGradientExhaustive
    // wiggles also the features and this leads to a gradient with respect to
    // the output. This is not what we want. We cannot get the gradient for the
    // features, yet.
    permutohedral_param->set_repeated_init(false);
    permutohedral_param->set_bias_term(true);
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    checker.CheckGradientExhaustive(&layer, bottom, top, 0);
  }

  static void TestGradientBlurSkip(const std::vector<Blob<Dtype>*>& bottom,
                                   const std::vector<Blob<Dtype>*>& top,
                                   int neighborhood_size,
                                   int num_output,
                                   int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);
    // Remark, it's important to set this to false! CheckGradientExhaustive
    // wiggles also the features and this leads to a gradient with respect to
    // the output. This is not what we want. We cannot get the gradient for the
    // features, yet.
    permutohedral_param->set_repeated_init(false);
    permutohedral_param->set_bias_term(false);
    permutohedral_param->set_do_skip_blur(true);

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    checker.CheckGradientExhaustive(&layer, bottom, top, 0);
  }

  static void TestGradientBiasBlurSkip(const std::vector<Blob<Dtype>*>& bottom,
                                       const std::vector<Blob<Dtype>*>& top,
                                       int neighborhood_size,
                                       int num_output,
                                       int group) {
    LayerParameter layer_param;
    PermutohedralParameter* permutohedral_param =
      layer_param.mutable_permutohedral_param();
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");
    permutohedral_param->set_neighborhood_size(neighborhood_size);
    permutohedral_param->set_num_output(num_output);
    permutohedral_param->set_group(group);
    // Remark, it's important to set this to false! CheckGradientExhaustive
    // wiggles also the features and this leads to a gradient with respect to
    // the output. This is not what we want. We cannot get the gradient for the
    // features, yet.
    permutohedral_param->set_repeated_init(false);
    permutohedral_param->set_bias_term(true);
    permutohedral_param->set_do_skip_blur(true);
    permutohedral_param->mutable_filter_filler()->set_type("gaussian");

    PermutohedralLayerTemplate<Dtype, Ptype> layer(layer_param);
    layer.SetUp(bottom, top);

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    checker.CheckGradientExhaustive(&layer, bottom, top, 0);
  }

 private:
  static Dtype kernel(const std::vector<Dtype>& left,
                      const std::vector<Dtype>& right) {
    CHECK_EQ(left.size(), right.size());
    Dtype dist = 0;
    for (int f = 0; f < left.size(); ++f) {
      const Dtype diff = left[f] - right[f];
      dist += diff * diff;
    }

    return std::exp(-0.5 * dist);
  }

  static void BruteForce(const int num_output,
                         const int group,
                         const std::vector<Blob<Dtype>*>& bottom,
                         std::vector<Blob<Dtype>*>* top) {
    top->resize(1);
    const Blob<Dtype>& data_blob         = *bottom[0];
    const Blob<Dtype>& in_features_blob  = *bottom[1];
    const Blob<Dtype>& out_features_blob = *bottom[2];

    const int num       = data_blob.num();
    const int channel   = data_blob.channels();
    const int in_height = data_blob.height();
    const int in_width  = data_blob.width();

    const int feature_size = in_features_blob.channels();

    const int out_height = out_features_blob.height();
    const int out_width  = out_features_blob.width();

    const int in_count  = in_height * in_width;
    const int out_count = out_height * out_width;

    CHECK_EQ(in_features_blob.num(), num);
    CHECK_EQ(in_features_blob.height(), in_height);
    CHECK_EQ(in_features_blob.width(), in_width);

    CHECK_EQ(out_features_blob.num(), num);
    CHECK_EQ(out_features_blob.channels(), feature_size);

    Blob<Dtype>& top_blob = *((*top)[0]);
    top_blob.Reshape(num, num_output, out_height, out_width);

    const Dtype* in           = data_blob.cpu_data();
    const Dtype* in_features  = in_features_blob.cpu_data();
    const Dtype* out_features = out_features_blob.cpu_data();
    Dtype* out                = top_blob.mutable_cpu_data();
    std::fill(out, out + num * num_output * out_count, 0);

    std::vector<Dtype> in_f(feature_size);
    std::vector<Dtype> out_f(feature_size);

    std::vector<Dtype> in_weight(num * in_count);
    std::vector<Dtype> out_weight(num * out_count);

    const int in_group_size  = channel / group;
    const int out_group_size = num_output / group;

    for (int n = 0; n < num; ++n) {
      Dtype k;
      // Precalculate the normalization constants.
      for (int i = 0; i < in_count; ++i) {
        for (int j = 0; j < out_count; ++j) {
          // Copy the featrues.
          for (int f = 0; f < feature_size; ++f) {
            in_f[f]  = in_features[(n * feature_size + f) * in_count + i];
            out_f[f] = out_features[(n * feature_size + f) * out_count + j];
          }

          k = kernel(in_f, out_f);
          out_weight[n * out_count + j] += k;
          in_weight[n * in_count + i] += k;
        }
      }

      // For every group do the convolution.
      for (int g = 0; g < group; ++g) {
        for (int i = 0; i < in_count; ++i) {
          for (int j = 0; j < out_count; ++j) {
            // Copy the featrues.
            for (int f = 0; f < feature_size; ++f) {
              in_f[f]  = in_features[(n * feature_size + f) * in_count + i];
              out_f[f] = out_features[(n * feature_size + f) * out_count + j];
            }

            for (int c = g * in_group_size;
                 c < std::min<int>((g + 1) * in_group_size, channel); ++c) {
              const Dtype influence = in[(n * channel + c) * in_count + i] /
                                      std::sqrt(in_weight[n * in_count + i]) /
                                      std::sqrt(out_weight[n * out_count + j]) *
                                      kernel(in_f, out_f);

              for (int o = g * out_group_size;
                   o < std::min<int>((g + 1) * out_group_size, num_output);
                   ++o) {
                out[(n * num_output + o) * out_count + j] += influence;
              }
            }
          }
        }
      }
    }
  }

  static void CheckBruteForce(const int num_output,
                              const int group,
                              const std::vector<Blob<Dtype>*>& bottom,
                              const std::vector<Blob<Dtype>*>& top) {
    // Create some extra blob to run for the check.
    std::vector<Blob<Dtype>*> model_top(1);
    model_top[0] = new Blob<Dtype>();

    BruteForce(num_output, group, bottom, &model_top);

    const Blob<Dtype>& top_blob       = *top[0];
    const Blob<Dtype>& model_top_blob = *model_top[0];

    const Dtype* top_data       = top_blob.cpu_data();
    const Dtype* model_top_data = model_top_blob.cpu_data();

    CHECK_EQ(top_blob.num(), model_top_blob.num());
    CHECK_EQ(top_blob.channels(), model_top_blob.channels());
    CHECK_EQ(top_blob.height(), model_top_blob.height());
    CHECK_EQ(top_blob.width(), model_top_blob.width());

    const int num     = top_blob.num();
    const int channel = top_blob.channels();
    const int count   = top_blob.height() * top_blob.width();

    Dtype serror = 0;
    Dtype max    = -std::numeric_limits<Dtype>::infinity();
    Dtype min    = std::numeric_limits<Dtype>::infinity();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channel; ++c) {
        for (int i = 0; i < count; ++i) {
          // std::cout << top_data[(n * channel + c) * count + i] << " ?= "
          //           << model_top_data[(n * channel + c) * count + i]
          //           << std::endl;
          const Dtype value    = top_data[(n * channel + c) * count + i];
          const Dtype expected = model_top_data[(n * channel + c) * count + i];
          max                  = std::max<Dtype>(expected, max);
          min                  = std::min<Dtype>(expected, min);
          const Dtype error    = value - expected;
          serror += error * error;
        }
      }
    }

    // Compute PSNR error.
    const Dtype psnr = 20 * std::log10(max - min) -
                       10 * std::log10(serror / (num * channel * count));
    std::cout << "PSNR: " << psnr << std::endl;
    CHECK_GT(psnr, 25);
  }
};

TYPED_TEST_CASE(PermutohedralLayerTest, TestDtypesAndDevices);

#define DEFINE_TYPED_TEST(test, type, neighborhood_size, num_output, group,    \
                          bottom)                                              \
  TYPED_TEST(                                                                  \
    PermutohedralLayerTest,                                                    \
    test##_##type##_##neighborhood_size##_##num_output##_##group##_##bottom) { \
    typedef typename TypeParam::Dtype Dtype;                                   \
    TestSuite<Dtype, permutohedral::type>::test(                               \
      this->bottom, this->top_, neighborhood_size, num_output, group);         \
  }

DEFINE_TYPED_TEST(TestSetup, Permutohedral, 3, 1, 1, bottom_)
DEFINE_TYPED_TEST(TestSetup, GaussPermutohedral, 2, 2, 2, big_bottom_)
DEFINE_TYPED_TEST(TestForward, Permutohedral, 3, 1, 1, big_bottom_)
DEFINE_TYPED_TEST(TestForward, Permutohedral, 3, 2, 1, big_bottom_)
DEFINE_TYPED_TEST(TestForward, Permutohedral, 3, 2, 2, big_bottom_)
DEFINE_TYPED_TEST(TestForward, GaussPermutohedral, 2, 2, 2, big_bottom_)
DEFINE_TYPED_TEST(TestForward, Permutohedral, 3, 1, 1, big_color_bottom_)
DEFINE_TYPED_TEST(TestForward, Permutohedral, 3, 2, 1, big_color_bottom_)
DEFINE_TYPED_TEST(TestForward, Permutohedral, 3, 2, 2, big_color_bottom_)
DEFINE_TYPED_TEST(TestForward, GaussPermutohedral, 2, 2, 2, big_color_bottom_)
DEFINE_TYPED_TEST(
  TestForwardDiagOffset, Permutohedral, 2, 2, 1, big_color_bottom_)
DEFINE_TYPED_TEST(TestGradient, Permutohedral, 3, 1, 1, bottom_)
DEFINE_TYPED_TEST(TestGradient, Permutohedral, 3, 2, 1, bottom_)
DEFINE_TYPED_TEST(TestGradient, Permutohedral, 3, 2, 2, bottom_)
DEFINE_TYPED_TEST(TestGradient, GaussPermutohedral, 2, 2, 2, bottom_)
DEFINE_TYPED_TEST(TestGradientBias, Permutohedral, 3, 1, 1, bottom_)
DEFINE_TYPED_TEST(TestGradientBias, Permutohedral, 3, 2, 1, bottom_)
DEFINE_TYPED_TEST(TestGradientBias, Permutohedral, 3, 2, 2, bottom_)
DEFINE_TYPED_TEST(TestGradientBias, Permutohedral, 3, 3, 2, bottom_)
DEFINE_TYPED_TEST(TestGradientBias, GaussPermutohedral, 2, 2, 2, bottom_)

DEFINE_TYPED_TEST(TestGradientBlurSkip, Permutohedral, 0, 2, 1, bottom_)
DEFINE_TYPED_TEST(TestGradientBlurSkip, Permutohedral, 0, 2, 2, bottom_)
DEFINE_TYPED_TEST(TestGradientBiasBlurSkip, Permutohedral, 0, 2, 1, bottom_)
DEFINE_TYPED_TEST(TestGradientBiasBlurSkip, Permutohedral, 0, 2, 2, bottom_)

DEFINE_TYPED_TEST(TestGradientBiasColor, Permutohedral, 2, 1, 1, color_bottom_)
DEFINE_TYPED_TEST(TestGradientBiasGroup, Permutohedral, 2, 2, 2, color_bottom_)
DEFINE_TYPED_TEST(
  TestGradientBiasGroup, GaussPermutohedral, 2, 2, 2, color_bottom_)

}  // namespace caffe
