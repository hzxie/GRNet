// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include <boost/array.hpp>
#include <boost/random.hpp>

#include <csignal>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/pixel_feature_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

/*
Setup function
*/
template <typename Dtype>
void PixelFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  ran_once = false;

  count_  = bottom[0]->count();
  num_    = bottom[0]->num();
  height_ = bottom[0]->height();
  width_  = bottom[0]->width();

  const PixelFeatureParameter& parameter =
    this->layer_param_.pixel_feature_param();

  switch (parameter.type()) {
    case PixelFeatureParameter_Feature_POSITION:
    case PixelFeatureParameter_Feature_RANDOM_POSITION:
    case PixelFeatureParameter_Feature_RANDOM_ROTATE: channels_ = 2; break;
    case PixelFeatureParameter_Feature_POSITION_AND_RGB:
      channels_ = 2 + bottom[0]->channels();
      break;
    case PixelFeatureParameter_Feature_RGB_AND_POSITION:
      channels_ = 2 + bottom[0]->channels();
      break;
    case PixelFeatureParameter_Feature_RGB:
      channels_ = bottom[0]->channels();
      break;
    case PixelFeatureParameter_Feature_WARPED_POSITION:
      // Location and the rotation angle.
      channels_ = 3;
      break;
    case PixelFeatureParameter_Feature_NUM_RANDOM_POSITION:
      height_   = 1;
      width_    = parameter.num_random_pos();
      channels_ = 2;
      break;
  }
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void PixelFeatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_, channels_, height_, width_);
}

/*
Forward CPU function
*/
template <typename Dtype>
void PixelFeatureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const PixelFeatureParameter& parameter =
    this->layer_param_.pixel_feature_param();

  Dtype* top_data = top[0]->mutable_cpu_data();

  switch (this->layer_param_.pixel_feature_param().type()) {
    case PixelFeatureParameter_Feature_POSITION: {
      if (!ran_once) {
        const Dtype scale    = parameter.pos_scale();
        const Dtype offset_h = parameter.offset_h();
        const Dtype offset_w = parameter.offset_w();

        for (unsigned int n = 0; n < num_; ++n) {
          for (unsigned int y = 0; y < height_; ++y) {
            Dtype y_offset = scale * y + offset_h;
            for (unsigned int x = 0; x < width_; ++x) {
              top_data[top[0]->offset(n, 0, y, x)] = y_offset;
              top_data[top[0]->offset(n, 1, y, x)] = scale * x + offset_w;
            }
          }
        }
      }

      break;
    }
    case PixelFeatureParameter_Feature_POSITION_AND_RGB: {
      const Dtype scale       = parameter.pos_scale();
      const Dtype color_scale = parameter.color_scale();
      const Dtype offset_h    = parameter.offset_h();
      const Dtype offset_w    = parameter.offset_w();

      for (unsigned int n = 0; n < num_; ++n) {
        for (unsigned int y = 0; y < height_; ++y) {
          Dtype y_offset = scale * y + offset_h;
          for (unsigned int x = 0; x < width_; ++x) {
            top_data[top[0]->offset(n, 0, y, x)] = y_offset;
            top_data[top[0]->offset(n, 1, y, x)] = scale * x + offset_w;
            for (unsigned int c = 0; c < bottom[0]->channels(); ++c) {
              top_data[top[0]->offset(n, c + 2, y, x)] =
                color_scale * bottom[0]->data_at(n, c, y, x);
            }
          }
        }
      }
      break;
    }
    case PixelFeatureParameter_Feature_RGB_AND_POSITION: {
      const Dtype scale       = parameter.pos_scale();
      const Dtype color_scale = parameter.color_scale();
      const Dtype offset_h    = parameter.offset_h();
      const Dtype offset_w    = parameter.offset_w();

      for (unsigned int n = 0; n < num_; ++n) {
        for (unsigned int y = 0; y < height_; ++y) {
          Dtype y_offset = scale * y + offset_h;
          for (unsigned int x = 0; x < width_; ++x) {
            for (unsigned int c = 0; c < bottom[0]->channels(); ++c) {
              top_data[top[0]->offset(n, c, y, x)] =
                color_scale * bottom[0]->data_at(n, c, y, x);
            }
            top_data[top[0]->offset(n, bottom[0]->channels(), y, x)] = y_offset;
            top_data[top[0]->offset(n, bottom[0]->channels() + 1, y, x)] =
              scale * x + offset_w;
          }
        }
      }
      break;
    }
    case PixelFeatureParameter_Feature_RGB: {
      const Dtype color_scale = parameter.color_scale();
      for (unsigned int n = 0; n < num_; ++n) {
        for (unsigned int y = 0; y < height_; ++y) {
          for (unsigned int x = 0; x < width_; ++x) {
            for (unsigned int c = 0; c < bottom[0]->channels(); ++c) {
              top_data[top[0]->offset(n, c, y, x)] =
                color_scale * bottom[0]->data_at(n, c, y, x);
            }
          }
        }
      }
      break;
    }
    case PixelFeatureParameter_Feature_RANDOM_POSITION:
    case PixelFeatureParameter_Feature_NUM_RANDOM_POSITION: {
      if (!ran_once) {
        const int input_height = bottom[0]->height();
        const int input_width  = bottom[0]->width();
        const Dtype scale      = parameter.pos_scale();

        boost::uniform_real<Dtype> random_height(0, input_height);
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
          variate_height(caffe_rng(), random_height);

        boost::uniform_real<Dtype> random_width(0, input_width);
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
          variate_width(caffe_rng(), random_width);

        for (unsigned int n = 0; n < num_; ++n) {
          for (unsigned int y = 0; y < height_; ++y) {
            for (unsigned int x = 0; x < width_; ++x) {
              top_data[top[0]->offset(n, 0, y, x)] = scale * variate_height();
              top_data[top[0]->offset(n, 1, y, x)] = scale * variate_width();
            }
          }
        }
      }
      break;
    }
    case PixelFeatureParameter_Feature_WARPED_POSITION: {
      if (!ran_once) {
        const Dtype angle       = -parameter.rotation_angle() / 180.0 * M_PI;
        const Dtype scale       = parameter.pos_scale();
        const Dtype angle_sigma = parameter.rotation_sigma();

        const Dtype cosAngle = std::cos(angle);
        const Dtype sinAngle = std::sin(angle);

        const Dtype mid_y = height_ / 2;
        const Dtype mid_x = width_ / 2;

        Dtype scaled_y, scaled_x;
        for (unsigned int n = 0; n < num_; ++n) {
          for (unsigned int y = 0; y < height_; ++y) {
            scaled_y = y * scale - mid_y;
            for (unsigned int x = 0; x < width_; ++x) {
              scaled_x = x * scale - mid_x;

              top_data[top[0]->offset(n, 0, y, x)] =
                sinAngle * scaled_x + cosAngle * scaled_y + mid_y;
              top_data[top[0]->offset(n, 1, y, x)] =
                cosAngle * scaled_x - sinAngle * scaled_y + mid_x;
              top_data[top[0]->offset(n, 2, y, x)] = angle / angle_sigma;
            }
          }
        }
      }
      break;
    }
    case PixelFeatureParameter_Feature_RANDOM_ROTATE: {
      if (!ran_once) {
        boost::uniform_real<Dtype> random_angle(-10, 10);
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
          variate_angle(caffe_rng(), random_angle);
        const Dtype angle = variate_angle() / 180.0 * M_PI;
        const Dtype scale = 1;

        const Dtype cosAngle = std::cos(angle);
        const Dtype sinAngle = std::sin(angle);

        const Dtype mid_y = height_ / 2;
        const Dtype mid_x = width_ / 2;

        Dtype scaled_y, scaled_x;
        for (unsigned int n = 0; n < num_; ++n) {
          for (unsigned int y = 0; y < height_; ++y) {
            scaled_y = y * scale - mid_y;
            for (unsigned int x = 0; x < width_; ++x) {
              scaled_x = x * scale - mid_x;

              top_data[top[0]->offset(n, 0, y, x)] =
                sinAngle * scaled_x + cosAngle * (scaled_y - mid_y) + mid_y;
              top_data[top[0]->offset(n, 1, y, x)] =
                cosAngle * scaled_x - sinAngle * scaled_y + mid_x;
            }
          }
        }
      }
      break;
    }
    default: LOG(FATAL) << "Undefined feature type of pixel feature layer";
  }

  ran_once = true;
}

/*
Backward CPU function
 */
template <typename Dtype>
void PixelFeatureLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {}

#ifdef CPU_ONLY
STUB_GPU(PixelFeatureLayer);
#endif

INSTANTIATE_CLASS(PixelFeatureLayer);
REGISTER_LAYER_CLASS(PixelFeature);

}  // namespace caffe
