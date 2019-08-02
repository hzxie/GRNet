// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#include <algorithm>
#include <csignal>
#include <iostream>
#include <string>
#include <vector>

#include "boost/array.hpp"
#include "boost/make_shared.hpp"

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/permutohedral_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  PermutohedralParameter permutohedral_param =
    this->layer_param_.permutohedral_param();

  do_repeated_init_ = permutohedral_param.repeated_init();
  do_visualization_ = permutohedral_param.visualize_lattice();
  do_skip_blur_     = permutohedral_param.do_skip_blur();

  neighborhood_size_ = permutohedral_param.neighborhood_size();
  num_output_        = permutohedral_param.num_output();
  group_             = permutohedral_param.group();

  bias_term_ = permutohedral_param.bias_term();

  Blob<Dtype>& data_blob        = *bottom[0];
  Blob<Dtype>& in_feature_blob  = *bottom[1];
  Blob<Dtype>& out_feature_blob = *bottom[2];

  num_       = data_blob.num();
  in_height_ = data_blob.height();
  in_width_  = data_blob.width();
  channels_  = data_blob.channels();

  feature_size_ = in_feature_blob.channels();

  out_height_ = out_feature_blob.height();
  out_width_  = out_feature_blob.width();

  CHECK_EQ(num_, in_feature_blob.num());
  CHECK_EQ(in_height_, in_feature_blob.height());
  CHECK_EQ(in_width_, in_feature_blob.width());

  CHECK_EQ(num_, out_feature_blob.num());
  CHECK_EQ(feature_size_, out_feature_blob.channels());

  // Use diagonal Gaussion option only if there is some kind of squared shape
  // dependency between input and output channels.
  if (permutohedral_param.offset_type() ==
      PermutohedralParameter_OffsetType_DIAG) {
    CHECK_EQ(num_output_, channels_ / group_)
      << "Diagonal offset can only be used when num_output = num_channels / "
         "group";
  }

  if (do_skip_blur_) {
    CHECK_EQ(channels_, num_output_)
      << "Number of output channels sould be same as input when skipping blur";
  }

  operations_.resize(num_);

  if (!this->blobs_.empty()) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(
        GetFiller<Dtype>(permutohedral_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    } else {
      this->blobs_.resize(1);
    }

    // Fill the filter weights with filter filler.
    this->blobs_[0].reset(new Blob<Dtype>(
      num_output_, channels_ / group_, 1,
      permutohedral_type::get_filter_size(neighborhood_size_, feature_size_)));

    shared_ptr<Filler<Dtype> > filter_filler(
      GetFiller<Dtype>(permutohedral_param.filter_filler()));
    filter_filler->Fill(this->blobs_[0].get());
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_, num_output_, out_height_, out_width_);

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    const int out_data_count = out_height_ * out_width_;

    bias_multiplier_.Reshape(1, 1, 1, out_data_count);
    caffe_set(out_data_count, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  if (top.size() > 1) {
    top[1]->Reshape(num_, 3, 1, sizeof(void*));
  }
}

template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::InitLattice(
  const Blob<Dtype>* const in_lattice_blob,
  const Blob<Dtype>& in_feature_blob,
  const Blob<Dtype>& out_feature_blob,
  Blob<Dtype>* const lattice_blob) {
  // We do a quick and dirty check if we've already initialized the
  // blurs. In case of the set configuration flag we do it again,
  // otherwise we skip it.
  if (!operations_.empty() && (!operations_[0].blur_ || do_repeated_init_)) {
    const int in_size    = in_height_ * in_width_;
    const int out_size   = out_height_ * out_width_;
    const int in_offset  = 0;
    const int out_offset = in_size;

    const int data_count = in_size + out_size;

    // Compute and initalize the lattices based on feature.
    const Dtype* in_feature_data  = in_feature_blob.cpu_data();
    const Dtype* out_feature_data = out_feature_blob.cpu_data();

    typename permutohedral_type::gauss_type gauss(neighborhood_size_,
                                                  feature_size_);
    const Dtype* gauss_filter = gauss.filter();

    Dtype* lattice_blob_data = 0;
    if (lattice_blob) lattice_blob_data = lattice_blob->mutable_cpu_data();
    const Dtype* in_lattice_blob_data = 0;
    if (in_lattice_blob) in_lattice_blob_data = in_lattice_blob->cpu_data();

#pragma omp parallel
    {
      std::vector<Dtype> feature(feature_size_ * data_count);

#pragma omp for
      for (int n = 0; n < num_; ++n) {
        for (int c = 0; c < feature_size_; ++c) {
          caffe_copy(in_size, in_feature_data + in_feature_blob.offset(n, c),
                     feature.data() + c * data_count);
          caffe_copy(out_size, out_feature_data + out_feature_blob.offset(n, c),
                     feature.data() + c * data_count + in_size);
        }

        BlurOperation& op = operations_[n];

        if (!in_lattice_blob_data) {
          op.blur_.reset(new permutohedral_type());
          op.blur_->init(feature.data(), data_count, feature_size_,
                         neighborhood_size_, do_visualization_);

          if (!op.norm_there_)
            op.norm_there_ =
              boost::make_shared<Blob<Dtype> >(1, 1, in_height_, in_width_);
          if (!op.norm_back_)
            op.norm_back_ =
              boost::make_shared<Blob<Dtype> >(1, 1, out_height_, out_width_);

          op.norm_there_->Reshape(1, 1, in_height_, in_width_);
          op.norm_back_->Reshape(1, 1, out_height_, out_width_);
          Dtype* norm_there_data = op.norm_there_->mutable_cpu_data();
          Dtype* norm_back_data  = op.norm_back_->mutable_cpu_data();

          // norm_type: PermutohedralParameter_NormType_INPUT: Non-symmetric
          // normalization using only input features
          // norm_typePermutohedralParameter_NormType_BOTH: symmetric
          // normalization using both input and output features
          std::fill(norm_there_data, norm_there_data + in_size, 1);

          switch (this->layer_param_.permutohedral_param().norm_type()) {
            case PermutohedralParameter_NormType_AFTER:
              op.blur_->compute(gauss_filter, norm_there_data, 1, 1, 1, false,
                                in_offset, out_offset, in_size, out_size,
                                norm_back_data);

              for (int i = 0; i < op.norm_back_->count(); ++i) {
                norm_back_data[i] = 1.0 / (norm_back_data[i] + 1e-20);
              }
              break;
            case PermutohedralParameter_NormType_SYMMETRIC:
              op.blur_->compute(gauss_filter, norm_there_data, 1, 1, 1, false,
                                in_offset, in_offset, in_size, in_size,
                                norm_there_data);

              std::fill(norm_back_data, norm_back_data + out_size, 1);
              op.blur_->compute(gauss_filter, norm_back_data, 1, 1, 1, false,
                                out_offset, out_offset, out_size, out_size,
                                norm_back_data);

              // TODO(mkiefel) this can also be done in parallel.
              for (int i = 0; i < op.norm_there_->count(); ++i) {
                norm_there_data[i] =
                  1.0 / (std::sqrt(norm_there_data[i] + 1e-20));
              }

              for (int i = 0; i < op.norm_back_->count(); ++i) {
                norm_back_data[i] =
                  1.0 / (std::sqrt(norm_back_data[i] + 1e-20));
              }
              break;
          }

          // This is an ugly hack to get the pointer to the newly initialized
          // permutohedral lattice to another layer. This assumes that there
          // is enough memory allocated in the lattice_blob.
          if (lattice_blob_data) {
            *reinterpret_cast<boost::shared_ptr<permutohedral_type>**>(
              lattice_blob_data + lattice_blob->offset(n, 0)) = &op.blur_;
            *reinterpret_cast<boost::shared_ptr<Blob<Dtype> >**>(
              lattice_blob_data + lattice_blob->offset(n, 1)) = &op.norm_there_;
            *reinterpret_cast<boost::shared_ptr<Blob<Dtype> >**>(
              lattice_blob_data + lattice_blob->offset(n, 2)) = &op.norm_back_;
          }
        } else {
          // We are given an initialized lattice, so we just copy over the
          // pointers.
          op.blur_ = **reinterpret_cast<
            const boost::shared_ptr<permutohedral_type>* const*>(
            in_lattice_blob_data + in_lattice_blob->offset(n, 0));
          op.norm_there_ =
            **reinterpret_cast<const boost::shared_ptr<Blob<Dtype> >* const*>(
              in_lattice_blob_data + in_lattice_blob->offset(n, 1));
          op.norm_back_ =
            **reinterpret_cast<const boost::shared_ptr<Blob<Dtype> >* const*>(
              in_lattice_blob_data + in_lattice_blob->offset(n, 2));
        }
      }
    }
  }
}

template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::OffsetFilter(
  const Blob<Dtype>& filter_blob, Blob<Dtype>* shifted_filter_blob) const {
  const Dtype* filter = filter_blob.cpu_data();

  // Add a standard Gaussian filter on top of the filter.
  // We only optimize the difference between the two!
  const int filter_size =
    permutohedral_type::get_filter_size(neighborhood_size_, feature_size_);
  shifted_filter_blob->Reshape(num_output_, channels_ / group_, 1, filter_size);
  Dtype* shifted_filter = shifted_filter_blob->mutable_cpu_data();

  // Get the current incarnation of a Gaussian filter.
  typename permutohedral_type::gauss_type gauss(neighborhood_size_,
                                                feature_size_);
  const Dtype* gauss_filter = gauss.filter();

  switch (this->layer_param_.permutohedral_param().offset_type()) {
    case PermutohedralParameter_OffsetType_FULL:
      for (int n = 0; n < num_output_; ++n) {
        for (int c = 0; c < channels_ / group_; ++c) {
          for (int i = 0; i < filter_size; ++i) {
            shifted_filter[shifted_filter_blob->offset(n, c, 0, i)] =
              filter[filter_blob.offset(n, c, 0, i)] + gauss_filter[i];
          }
        }
      }
      break;
    case PermutohedralParameter_OffsetType_DIAG:
      std::copy(filter, filter + shifted_filter_blob->count(), shifted_filter);
      for (int n = 0; n < num_output_; ++n) {
        for (int i = 0; i < filter_size; ++i) {
          shifted_filter[shifted_filter_blob->offset(n, n, 0, i)] +=
            gauss_filter[i];
        }
      }
      break;
    case PermutohedralParameter_OffsetType_NONE:
      std::copy(filter, filter + filter_blob.count(), shifted_filter);
      break;
  }
}

template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& in_feature_blob  = *bottom[1];
  const Blob<Dtype>& out_feature_blob = *bottom[2];

  const Blob<Dtype>* in_lattice_blob = 0;
  if (bottom.size() > 3) {
    in_lattice_blob = bottom[3];
  }
  Blob<Dtype>* lattice_blob = 0;
  if (top.size() > 1) {
    lattice_blob = top[1];
  }

  InitLattice(in_lattice_blob, in_feature_blob, out_feature_blob, lattice_blob);

  const Blob<Dtype>& bottom_blob = *bottom[0];
  const Dtype* bottom_data       = bottom_blob.cpu_data();
  Blob<Dtype>& top_blob          = *top[0];
  Dtype* top_data                = top_blob.mutable_cpu_data();

  const Blob<Dtype>& filter_blob = *this->blobs_[0];
  Blob<Dtype> shifted_filter_blob;
  OffsetFilter(filter_blob, &shifted_filter_blob);
  const Dtype* const shifted_filter = shifted_filter_blob.cpu_data();

  const int in_size    = in_height_ * in_width_;
  const int out_size   = out_height_ * out_width_;
  const int in_offset  = 0;
  const int out_offset = in_size;

  const int out_data_count = out_height_ * out_width_;

  caffe_set<Dtype>(top_blob.count(), 0, top_data);

#pragma omp parallel
  {
    Blob<Dtype> scaled_there;
    scaled_there.Reshape(1, channels_, in_height_, in_width_);
    Dtype* scaled_there_data = scaled_there.mutable_cpu_data();

#pragma omp for
    for (int n = 0; n < num_; ++n) {
      BlurOperation& op = operations_[n];

      const Dtype* norm_there_data = op.norm_there_->cpu_data();
      const Dtype* norm_back_data  = op.norm_back_->cpu_data();

      for (int c = 0; c < channels_; ++c) {
        caffe_mul(in_size, norm_there_data,
                  bottom_data + bottom_blob.offset(n, c),
                  scaled_there_data + scaled_there.offset(0, c));
      }

      // Compute the permutohedral filter response
      op.reverse_ = op.blur_->compute(
        shifted_filter, scaled_there_data, num_output_, group_, channels_,
        do_skip_blur_, in_offset, out_offset, in_size, out_size,
        top_data + top_blob.offset(n));

      for (int c = 0; c < num_output_; ++c) {
        caffe_mul(out_size, top_data + top_blob.offset(n, c), norm_back_data,
                  top_data + top_blob.offset(n, c));
      }

      // Add bias.
      if (bias_term_) {
        caffe_cpu_gemm<Dtype>(
          CblasNoTrans, CblasNoTrans, num_output_, out_data_count, 1, (Dtype)1.,
          this->blobs_[1]->cpu_data(), bias_multiplier_.cpu_data(), (Dtype)1.,
          top_data + top_blob.offset(n));
      }
    }
  }
}

template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype>& filter_blob = *this->blobs_[0];

  Dtype* filter_diff = filter_blob.mutable_cpu_diff();
  caffe_set<Dtype>(filter_blob.count(), 0, filter_diff);

  const Blob<Dtype>& top_blob = *top[0];
  const Dtype* top_diff       = top_blob.cpu_diff();

  Blob<Dtype>& bottom_blob = *bottom[0];
  Dtype* bottom_diff       = bottom_blob.mutable_cpu_diff();

  caffe_set<Dtype>(bottom_blob.count(), 0, bottom_diff);

  // Gradient with respect to bias.
  if (bias_term_) {
    Blob<Dtype>& bias_blob = *this->blobs_[1];
    Dtype* bias_diff       = bias_blob.mutable_cpu_diff();
    caffe_set<Dtype>(bias_blob.count(), 0, bias_diff);

    const int out_data_count = out_height_ * out_width_;

    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_data_count, 1.,
                            top_diff + top_blob.offset(n),
                            bias_multiplier_.cpu_data(), 1., bias_diff);
    }
  }

  const int in_size  = in_height_ * in_width_;
  const int out_size = out_height_ * out_width_;

// Gradient computation.
#pragma omp parallel
  {
    Blob<Dtype> scaled_back;
    scaled_back.Reshape(1, num_output_, out_height_, out_width_);
    Dtype* scaled_back_data = scaled_back.mutable_cpu_data();

    Blob<Dtype> thread_filter_blob(filter_blob.num(), filter_blob.channels(),
                                   filter_blob.height(), filter_blob.width());
    Dtype* thread_filter_diff = thread_filter_blob.mutable_cpu_diff();
    caffe_set<Dtype>(thread_filter_blob.count(), 0, thread_filter_diff);

#pragma omp for
    for (int n = 0; n < num_; ++n) {
      BlurOperation& op = operations_[n];

      const Dtype* norm_there_data = op.norm_there_->cpu_data();
      const Dtype* norm_back_data  = op.norm_back_->cpu_data();

      for (int c = 0; c < num_output_; ++c) {
        caffe_mul(out_size, top_diff + top_blob.offset(n, c), norm_back_data,
                  scaled_back_data + scaled_back.offset(0, c));
      }

      op.reverse_->reverse(scaled_back_data, thread_filter_diff,
                           bottom_diff + bottom_blob.offset(n));

      for (int c = 0; c < channels_; ++c) {
        caffe_mul(in_size, bottom_diff + bottom_blob.offset(n, c),
                  norm_there_data, bottom_diff + bottom_blob.offset(n, c));
      }
    }

#pragma omp critical
    {
      for (int f = 0; f < thread_filter_blob.count(); ++f) {
        filter_diff[f] += thread_filter_diff[f];
      }
    }
  }
}

#ifdef CPU_ONLY
template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->Forward_cpu(bottom, top);
}

template <typename Dtype, template <typename> class Ptype>
void PermutohedralLayerTemplate<Dtype, Ptype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  this->Backward_cpu(top, propagate_down, bottom);
}
#endif

template class PermutohedralLayerTemplate<float, permutohedral::Permutohedral>;
template class PermutohedralLayerTemplate<double, permutohedral::Permutohedral>;
template class PermutohedralLayerTemplate<float,
                                          permutohedral::GaussPermutohedral>;
template class PermutohedralLayerTemplate<double,
                                          permutohedral::GaussPermutohedral>;

template <typename Dtype>
shared_ptr<Layer<Dtype> > Creator_PermutohedralLayer(
  const LayerParameter& param) {
  return shared_ptr<Layer<Dtype> >(
    new PermutohedralLayerTemplate<Dtype, permutohedral::Permutohedral>(param));
}

template <typename Dtype>
shared_ptr<Layer<Dtype> > Creator_GaussPermutohedralLayer(
  const LayerParameter& param) {
  return shared_ptr<Layer<Dtype> >(
    new PermutohedralLayerTemplate<Dtype, permutohedral::GaussPermutohedral>(
      param));
}

REGISTER_LAYER_CREATOR(GaussPermutohedral, Creator_GaussPermutohedralLayer);
REGISTER_LAYER_CREATOR(Permutohedral, Creator_PermutohedralLayer);

}  // namespace caffe
