// Copyright 2015 MPI Tuebingen

#include <algorithm>
#include <csignal>
#include <iostream>
#include <string>
#include <vector>

#include "boost/array.hpp"
#include "boost/make_shared.hpp"

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/permutohedral_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  PermutohedralPoolingParameter permutohedral_pooling_param =
    this->layer_param_.permutohedral_pooling_param();

  do_repeated_init_  = permutohedral_pooling_param.repeated_init();
  neighborhood_size_ = permutohedral_pooling_param.neighborhood_size();

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

  operations_.resize(num_);
}

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_, channels_, out_height_, out_width_);

  if (top.size() > 1) {
    top[1]->Reshape(num_, 3, 1, sizeof(void*));
  }
}

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::InitLattice(
  const Blob<Dtype>* const in_lattice_blob,
  const Blob<Dtype>& in_feature_blob,
  const Blob<Dtype>& out_feature_blob,
  Blob<Dtype>* const lattice_blob) {
  // We do a quick and dirty check if we've already initialized the
  // blurs. In case of the set configuration flag we do it again,
  // otherwise we skip it.
  if (!operations_.empty() && (!operations_[0].max_ || do_repeated_init_)) {
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

        MaxOperation& op = operations_[n];

        if (!in_lattice_blob_data) {
          op.max_.reset(new permutohedral_type());
          op.max_->init(feature.data(), data_count, feature_size_,
                        neighborhood_size_, false);

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

          // norm_type: PermutohedralPoolingParameter_NormType_INPUT:
          // Non-symmetric normalization using only input features
          // norm_typePermutohedralPoolingParameter_NormType_BOTH: symmetric
          // normalization using both input and output features
          std::fill(norm_there_data, norm_there_data + in_size, 1);

          switch (
            this->layer_param_.permutohedral_pooling_param().norm_type()) {
            case PermutohedralPoolingParameter_NormType_AFTER:
              op.max_->max_compute(gauss_filter, norm_there_data, 1, in_offset,
                                   out_offset, in_size, out_size,
                                   norm_back_data);

              for (int i = 0; i < op.norm_back_->count(); ++i) {
                norm_back_data[i] = 1.0 / (norm_back_data[i] + 1e-20);
              }
              break;
            case PermutohedralPoolingParameter_NormType_SYMMETRIC:
              op.max_->max_compute(gauss_filter, norm_there_data, 1, in_offset,
                                   in_offset, in_size, in_size,
                                   norm_there_data);

              std::fill(norm_back_data, norm_back_data + out_size, 1);
              op.max_->max_compute(gauss_filter, norm_back_data, 1, out_offset,
                                   out_offset, out_size, out_size,
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
              lattice_blob_data + lattice_blob->offset(n, 0)) = &op.max_;
            *reinterpret_cast<boost::shared_ptr<Blob<Dtype> >**>(
              lattice_blob_data + lattice_blob->offset(n, 1)) = &op.norm_there_;
            *reinterpret_cast<boost::shared_ptr<Blob<Dtype> >**>(
              lattice_blob_data + lattice_blob->offset(n, 2)) = &op.norm_back_;
          }
        } else {
          // We are given an initialized lattice, so we just copy over the
          // pointers.
          op.max_ = **reinterpret_cast<
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

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::Forward_cpu(
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

  typename permutohedral_type::gauss_type gauss(neighborhood_size_,
                                                feature_size_);
  const Dtype* gauss_filter = gauss.filter();

  const int in_size    = in_height_ * in_width_;
  const int out_size   = out_height_ * out_width_;
  const int in_offset  = 0;
  const int out_offset = in_size;

  caffe_set<Dtype>(top_blob.count(), 0, top_data);

#pragma omp parallel
  {
    Blob<Dtype> scaled_there;
    scaled_there.Reshape(1, channels_, in_height_, in_width_);
    Dtype* scaled_there_data = scaled_there.mutable_cpu_data();

#pragma omp for
    for (int n = 0; n < num_; ++n) {
      MaxOperation& op = operations_[n];

      const Dtype* norm_there_data = op.norm_there_->cpu_data();
      const Dtype* norm_back_data  = op.norm_back_->cpu_data();

      for (int c = 0; c < channels_; ++c) {
        caffe_mul(in_size, norm_there_data,
                  bottom_data + bottom_blob.offset(n, c),
                  scaled_there_data + scaled_there.offset(0, c));
      }

      // Compute the permutohedral max response
      op.reverse_ = op.max_->max_compute(
        gauss_filter, scaled_there_data, channels_, in_offset, out_offset,
        in_size, out_size, top_data + top_blob.offset(n));

      for (int c = 0; c < channels_; ++c) {
        caffe_mul(out_size, top_data + top_blob.offset(n, c), norm_back_data,
                  top_data + top_blob.offset(n, c));
      }
    }
  }
}

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  const Blob<Dtype>& top_blob = *top[0];
  const Dtype* top_diff       = top_blob.cpu_diff();

  Blob<Dtype>& bottom_blob = *bottom[0];
  Dtype* bottom_diff       = bottom_blob.mutable_cpu_diff();

  caffe_set<Dtype>(bottom_blob.count(), 0, bottom_diff);

  const int in_size  = in_height_ * in_width_;
  const int out_size = out_height_ * out_width_;

// Gradient computation.
#pragma omp parallel
  {
    Blob<Dtype> scaled_back;
    scaled_back.Reshape(1, channels_, out_height_, out_width_);
    Dtype* scaled_back_data = scaled_back.mutable_cpu_data();

#pragma omp for
    for (int n = 0; n < num_; ++n) {
      MaxOperation& op = operations_[n];

      const Dtype* norm_there_data = op.norm_there_->cpu_data();
      const Dtype* norm_back_data  = op.norm_back_->cpu_data();

      for (int c = 0; c < channels_; ++c) {
        caffe_mul(out_size, top_diff + top_blob.offset(n, c), norm_back_data,
                  scaled_back_data + scaled_back.offset(0, c));
      }

      op.reverse_->max_reverse(scaled_back_data,
                               bottom_diff + bottom_blob.offset(n));

      for (int c = 0; c < channels_; ++c) {
        caffe_mul(in_size, bottom_diff + bottom_blob.offset(n, c),
                  norm_there_data, bottom_diff + bottom_blob.offset(n, c));
      }
    }
  }
}

#ifdef CPU_ONLY
template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  this->Backward_cpu(top, propagate_down, bottom);
}
#endif

INSTANTIATE_CLASS(PermutohedralPoolingLayer);

}  // namespace caffe
