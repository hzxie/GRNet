// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include <boost/array.hpp>
#include <boost/random.hpp>

#include "caffe/layers/pixel_feature_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PixelFeatureXYForwardGPU(const int nthreads,
                                         const int height,
                                         const int width,
                                         const Dtype pos_scale,
                                         const Dtype offset_h,
                                         const Dtype offset_w,
                                         Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int spatial_dim = height * width;
    const int n           = index / spatial_dim;
    const int s           = index % spatial_dim;

    const int y = s / width;
    const int x = s % width;

    int out_dim            = 2;
    int top_offset_1       = ((n * out_dim) * spatial_dim + s);
    int top_offset_2       = ((n * out_dim + 1) * spatial_dim + s);
    top_data[top_offset_1] = pos_scale * y + offset_h;
    top_data[top_offset_2] = pos_scale * x + offset_w;
  }
}

template <typename Dtype>
__global__ void PixelFeatureXYRGBForwardGPU(const int nthreads,
                                            const Dtype* bottom_data,
                                            const int height,
                                            const int width,
                                            const int in_dim,
                                            const Dtype pos_scale,
                                            const Dtype color_scale,
                                            const Dtype offset_h,
                                            const Dtype offset_w,
                                            Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int spatial_dim = height * width;
    const int n           = index / spatial_dim;
    const int s           = index % spatial_dim;

    const int y = s / width;
    const int x = s % width;

    int out_dim            = 2 + in_dim;
    int top_offset_1       = ((n * out_dim) * spatial_dim + s);
    int top_offset_2       = ((n * out_dim + 1) * spatial_dim + s);
    top_data[top_offset_1] = pos_scale * y + offset_h;
    top_data[top_offset_2] = pos_scale * x + offset_w;

    for (unsigned int c = 0; c < in_dim; ++c) {
      int bottom_offset    = ((n * in_dim + c) * spatial_dim + s);
      int top_offset       = ((n * out_dim + c + 2) * spatial_dim + s);
      top_data[top_offset] = color_scale * bottom_data[bottom_offset];
    }
  }
}

template <typename Dtype>
__global__ void PixelFeatureRGBXYForwardGPU(const int nthreads,
                                            const Dtype* bottom_data,
                                            const int height,
                                            const int width,
                                            const int in_dim,
                                            const Dtype pos_scale,
                                            const Dtype color_scale,
                                            const Dtype offset_h,
                                            const Dtype offset_w,
                                            Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int spatial_dim = height * width;
    const int n           = index / spatial_dim;
    const int s           = index % spatial_dim;

    const int y = s / width;
    const int x = s % width;

    int out_dim = 2 + in_dim;

    for (unsigned int c = 0; c < in_dim; ++c) {
      int bottom_offset    = ((n * in_dim + c) * spatial_dim + s);
      int top_offset       = ((n * out_dim + c) * spatial_dim + s);
      top_data[top_offset] = color_scale * bottom_data[bottom_offset];
    }

    int top_offset_1       = ((n * out_dim + in_dim) * spatial_dim + s);
    int top_offset_2       = ((n * out_dim + in_dim + 1) * spatial_dim + s);
    top_data[top_offset_1] = pos_scale * y + offset_h;
    top_data[top_offset_2] = pos_scale * x + offset_w;
  }
}

template <typename Dtype>
__global__ void PixelFeatureRGBForwardGPU(const int nthreads,
                                          const Dtype* bottom_data,
                                          const int height,
                                          const int width,
                                          const int in_dim,
                                          const Dtype color_scale,
                                          Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int spatial_dim = height * width;
    const int n           = index / spatial_dim;
    const int s           = index % spatial_dim;

    int out_dim = in_dim;
    for (unsigned int c = 0; c < in_dim; ++c) {
      int bottom_offset    = ((n * in_dim + c) * spatial_dim + s);
      int top_offset       = ((n * out_dim + c) * spatial_dim + s);
      top_data[top_offset] = color_scale * bottom_data[bottom_offset];
    }
  }
}

template <typename Dtype>
__global__ void PixelFeatureWARPPOSForwardGPU(const int nthreads,
                                              const int height,
                                              const int width,
                                              const Dtype pos_scale,
                                              const Dtype angle,
                                              const Dtype angle_sigma,
                                              const Dtype cosAngle,
                                              const Dtype sinAngle,
                                              Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int spatial_dim = height * width;
    const int n           = index / spatial_dim;
    const int s           = index % spatial_dim;

    const int y = s / width;
    const int x = s % width;

    const Dtype mid_y = height / 2;
    const Dtype mid_x = width / 2;

    const Dtype scaled_y = pos_scale * y;
    const Dtype scaled_x = pos_scale * x;

    int out_dim      = 3;
    int top_offset_1 = ((n * out_dim) * spatial_dim + s);
    int top_offset_2 = ((n * out_dim + 1) * spatial_dim + s);
    int top_offset_3 = ((n * out_dim + 2) * spatial_dim + s);

    top_data[top_offset_1] =
      sinAngle * (scaled_x - mid_x) + cosAngle * (scaled_y - mid_y) + mid_y;
    top_data[top_offset_2] =
      cosAngle * (scaled_x - mid_x) - sinAngle * (scaled_y - mid_y) + mid_x;
    top_data[top_offset_3] = angle / angle_sigma;
  }
}

template <typename Dtype>
__global__ void PixelFeatureRANDROTATEForwardGPU(const int nthreads,
                                                 const int height,
                                                 const int width,
                                                 const Dtype pos_scale,
                                                 const Dtype cosAngle,
                                                 const Dtype sinAngle,
                                                 Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int spatial_dim = height * width;
    const int n           = index / spatial_dim;
    const int s           = index % spatial_dim;

    const int y = s / width;
    const int x = s % width;

    const Dtype mid_y = height / 2;
    const Dtype mid_x = width / 2;

    const Dtype scaled_y = pos_scale * y;
    const Dtype scaled_x = pos_scale * x;

    int out_dim      = 2;
    int top_offset_1 = ((n * out_dim) * spatial_dim + s);
    int top_offset_2 = ((n * out_dim + 1) * spatial_dim + s);

    top_data[top_offset_1] =
      sinAngle * (scaled_x - mid_x) + cosAngle * (scaled_y - mid_y) + mid_y;
    top_data[top_offset_2] =
      cosAngle * (scaled_x - mid_x) - sinAngle * (scaled_y - mid_y) + mid_x;
  }
}

/*
Forward CPU function
*/
template <typename Dtype>
void PixelFeatureLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const PixelFeatureParameter& parameter =
    this->layer_param_.pixel_feature_param();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data          = top[0]->mutable_gpu_data();

  switch (this->layer_param_.pixel_feature_param().type()) {
    case PixelFeatureParameter_Feature_POSITION: {
      if (!ran_once) {
        const Dtype scale    = parameter.pos_scale();
        const Dtype offset_h = parameter.offset_h();
        const Dtype offset_w = parameter.offset_w();

        const int nthreads = num_ * height_ * width_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        PixelFeatureXYForwardGPU<Dtype>
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, height_, width_, scale, offset_h, offset_w, top_data);
      }
      break;
    }
    case PixelFeatureParameter_Feature_POSITION_AND_RGB: {
      const Dtype scale       = parameter.pos_scale();
      const Dtype color_scale = parameter.color_scale();
      const Dtype offset_h    = parameter.offset_h();
      const Dtype offset_w    = parameter.offset_w();

      const int nthreads = num_ * height_ * width_;
      const int channels = bottom[0]->channels();
      // NOLINT_NEXT_LINE(whitespace/operators)
      PixelFeatureXYRGBForwardGPU<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, bottom_data, height_, width_, channels, scale, color_scale,
          offset_h, offset_w, top_data);
      break;
    }
    case PixelFeatureParameter_Feature_RGB_AND_POSITION: {
      const Dtype scale       = parameter.pos_scale();
      const Dtype color_scale = parameter.color_scale();
      const Dtype offset_h    = parameter.offset_h();
      const Dtype offset_w    = parameter.offset_w();

      const int nthreads = num_ * height_ * width_;
      const int channels = bottom[0]->channels();
      // NOLINT_NEXT_LINE(whitespace/operators)
      PixelFeatureRGBXYForwardGPU<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, bottom_data, height_, width_, channels, scale, color_scale,
          offset_h, offset_w, top_data);
      break;
    }
    case PixelFeatureParameter_Feature_RGB: {
      const Dtype color_scale = parameter.color_scale();
      const int nthreads      = num_ * height_ * width_;
      const int channels      = bottom[0]->channels();
      // NOLINT_NEXT_LINE(whitespace/operators)
      PixelFeatureRGBForwardGPU<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, bottom_data, height_, width_, channels, color_scale,
          top_data);
      break;
    }
    case PixelFeatureParameter_Feature_RANDOM_POSITION:
    case PixelFeatureParameter_Feature_NUM_RANDOM_POSITION: {
      if (!ran_once) {
        const int input_height = bottom[0]->height();
        const int input_width  = bottom[0]->width();
        const Dtype scale      = parameter.pos_scale();

        boost::uniform_real<Dtype> random_height(0, input_height);
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype>>
          variate_height(caffe_rng(), random_height);

        boost::uniform_real<Dtype> random_width(0, input_width);
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype>>
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

        const int nthreads = num_ * height_ * width_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        PixelFeatureWARPPOSForwardGPU<Dtype>
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, height_, width_, scale, angle, angle_sigma, cosAngle,
            sinAngle, top_data);
      }
      break;
    }
    case PixelFeatureParameter_Feature_RANDOM_ROTATE: {
      if (!ran_once) {
        boost::uniform_real<Dtype> random_angle(-10, 10);
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype>>
          variate_angle(caffe_rng(), random_angle);
        const Dtype angle = variate_angle() / 180.0 * M_PI;
        const Dtype scale = 1;

        const Dtype cosAngle = std::cos(angle);
        const Dtype sinAngle = std::sin(angle);

        const int nthreads = num_ * height_ * width_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        PixelFeatureRANDROTATEForwardGPU<Dtype>
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, height_, width_, scale, cosAngle, sinAngle, top_data);
      }
      break;
    }
    default: LOG(FATAL) << "Undefined feature type of pixel feature layer";
  }

  ran_once = true;
}

/*
Backward GPU function
 */
template <typename Dtype>
void PixelFeatureLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {}

INSTANTIATE_LAYER_GPU_FUNCS(PixelFeatureLayer);

}  // namespace caffe
