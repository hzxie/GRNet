// Copyright 2019 Haozhe Xie
// Distributed under the MIT Software license,
// (See https://opensource.org/licenses/MIT)

#include <vector>

#include "caffe/layers/chamfer_distance_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChamferDistanceLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), 3);
  CHECK_EQ(bottom[1]->height(), 3);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void ChamferDistanceLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  dist1_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist2_.Reshape(bottom[1]->num(), bottom[1]->channels(), 1, 1);
  indexes1_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  indexes2_.Reshape(bottom[1]->num(), bottom[1]->channels(), 1, 1);
}

template <typename Dtype>
void ChamferDistanceLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* m_dist1 = dist1_.mutable_cpu_data();
  Dtype* m_dist2 = dist2_.mutable_cpu_data();
  int* indexes1  = indexes1_.mutable_cpu_data();
  int* indexes2  = indexes2_.mutable_cpu_data();

  const int num       = bottom[0]->num();
  const int n_points1 = bottom[0]->channels();
  const int n_points2 = bottom[1]->channels();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < n_points1; ++j) {
      Dtype _min_distance(1e12);
      for (int k = 0; k < n_points2; ++k) {
        Dtype x1   = bottom[0]->data_at(i, j, 0, 0);
        Dtype y1   = bottom[0]->data_at(i, j, 1, 0);
        Dtype z1   = bottom[0]->data_at(i, j, 2, 0);
        Dtype x2   = bottom[1]->data_at(i, k, 0, 0);
        Dtype y2   = bottom[1]->data_at(i, k, 1, 0);
        Dtype z2   = bottom[1]->data_at(i, k, 2, 0);
        Dtype dx   = x1 - x2;
        Dtype dy   = y1 - y2;
        Dtype dz   = z1 - z2;
        Dtype dist = dx * dx + dy * dy + dz * dz;
        if (dist < _min_distance) {
          _min_distance               = dist;
          indexes1[i * n_points1 + j] = i * n_points2 + k;
        }
      }
      m_dist1[i * n_points1 + j] += _min_distance / n_points1;
    }

    for (int j = 0; j < n_points2; ++j) {
      Dtype _min_distance(1e12);
      for (int k = 0; k < n_points1; ++k) {
        Dtype x1   = bottom[1]->data_at(i, j, 0, 0);
        Dtype y1   = bottom[1]->data_at(i, j, 1, 0);
        Dtype z1   = bottom[1]->data_at(i, j, 2, 0);
        Dtype x2   = bottom[0]->data_at(i, k, 0, 0);
        Dtype y2   = bottom[0]->data_at(i, k, 1, 0);
        Dtype z2   = bottom[0]->data_at(i, k, 2, 0);
        Dtype dx   = x1 - x2;
        Dtype dy   = y1 - y2;
        Dtype dz   = z1 - z2;
        Dtype dist = dx * dx + dy * dy + dz * dz;
        if (dist < _min_distance) {
          _min_distance               = dist;
          indexes2[i * n_points2 + j] = i * n_points1 + k;
        }
      }
      m_dist2[i * n_points2 + j] += _min_distance / n_points2;
    }
  }

  Dtype loss1 = caffe_cpu_asum(dist1_.count(), dist1_.cpu_data()) / num;
  Dtype loss2 = caffe_cpu_asum(dist2_.count(), dist2_.cpu_data()) / num;
  Dtype loss  = loss1 + loss2;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ChamferDistanceLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {}

#ifdef CPU_ONLY
STUB_GPU(ChamferDistanceLossLayer);
#endif

INSTANTIATE_CLASS(ChamferDistanceLossLayer);
REGISTER_LAYER_CLASS(ChamferDistanceLoss);

}  // namespace caffe
