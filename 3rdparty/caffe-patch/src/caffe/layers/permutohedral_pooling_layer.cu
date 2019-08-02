// Copyright 2015 MPI Tuebingen

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/permutohedral_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::Forward_gpu(
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
  const Dtype* bottom_data       = bottom_blob.gpu_data();
  Blob<Dtype>& top_blob          = *top[0];
  Dtype* top_data                = top_blob.mutable_gpu_data();

  typename permutohedral_type::gauss_type gauss(neighborhood_size_,
                                                feature_size_);
  const Dtype* gauss_filter = gauss.filter();

  const int in_size    = in_height_ * in_width_;
  const int out_size   = out_height_ * out_width_;
  const int in_offset  = 0;
  const int out_offset = in_size;

  caffe_gpu_memset(top_blob.count() * sizeof(Dtype), 0, top_data);

  Blob<Dtype> scaled_there;
  scaled_there.Reshape(1, channels_, in_height_, in_width_);

  Dtype* const scaled_there_data = scaled_there.mutable_gpu_data();

  for (int n = 0; n < num_; ++n) {
    MaxOperation& op = operations_[n];

    const Dtype* const norm_there_data = op.norm_there_->gpu_data();
    const Dtype* const norm_back_data  = op.norm_back_->gpu_data();

    for (int c = 0; c < channels_; ++c) {
      caffe_gpu_mul(in_size, norm_there_data,
                    bottom_data + bottom_blob.offset(n, c),
                    scaled_there_data + scaled_there.offset(0, c));
    }

    // Compute the permutohedral filter response
    op.reverse_ = op.max_->max_compute_gpu(
      gauss_filter, scaled_there_data, channels_, in_offset, out_offset,
      in_size, out_size, top_data + top_blob.offset(n));

    for (int c = 0; c < channels_; ++c) {
      caffe_gpu_mul(out_size, top_data + top_blob.offset(n, c), norm_back_data,
                    top_data + top_blob.offset(n, c));
    }
  }
}

template <typename Dtype>
void PermutohedralPoolingLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  const Blob<Dtype>& top_blob = *top[0];
  const Dtype* const top_diff = top_blob.gpu_diff();

  Blob<Dtype>& bottom_blob = *bottom[0];
  Dtype* const bottom_diff = bottom_blob.mutable_gpu_diff();

  caffe_gpu_memset(bottom_blob.count() * sizeof(Dtype), 0, bottom_diff);

  // Gradient computation.
  Blob<Dtype> scaled_back;
  scaled_back.Reshape(1, channels_, out_height_, out_width_);

  Dtype* scaled_back_data = scaled_back.mutable_gpu_data();

  const int in_size  = in_height_ * in_width_;
  const int out_size = out_height_ * out_width_;

  for (int n = 0; n < num_; ++n) {
    MaxOperation& op = operations_[n];

    const Dtype* norm_there_data = op.norm_there_->gpu_data();
    const Dtype* norm_back_data  = op.norm_back_->gpu_data();

    for (int c = 0; c < channels_; ++c) {
      caffe_gpu_mul(out_size, top_diff + top_blob.offset(n, c), norm_back_data,
                    scaled_back_data + scaled_back.offset(0, c));
    }

    op.reverse_->max_reverse(scaled_back_data,
                             bottom_diff + bottom_blob.offset(n));

    for (int c = 0; c < channels_; ++c) {
      caffe_gpu_mul(in_size, bottom_diff + bottom_blob.offset(n, c),
                    norm_there_data, bottom_diff + bottom_blob.offset(n, c));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PermutohedralPoolingLayer);

}  // namespace caffe
