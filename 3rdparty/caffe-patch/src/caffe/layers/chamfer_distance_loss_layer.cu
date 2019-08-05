// Copyright 2019 Haozhe Xie
// Distributed under the MIT Software license,
// (See https://opensource.org/licenses/MIT)

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "caffe/layers/chamfer_distance_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void NmDistanceKernel(const int num,
                                 const int n_points1,
                                 const Dtype* points1,
                                 const int n_points2,
                                 const Dtype* points2,
                                 Dtype* diff,
                                 int* indexes) {
  __shared__ Dtype buf[CAFFE_CUDA_NUM_THREADS * 3];
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    for (int k2 = 0; k2 < n_points2; k2 += CAFFE_CUDA_NUM_THREADS) {
      int end_k = min(n_points2, k2 + CAFFE_CUDA_NUM_THREADS) - k2;
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = points2[(i * n_points2 + k2) * 3 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n_points1;
           j += blockDim.x * gridDim.y) {
        Dtype x1   = points1[(i * n_points1 + j) * 3 + 0];
        Dtype y1   = points1[(i * n_points1 + j) * 3 + 1];
        Dtype z1   = points1[(i * n_points1 + j) * 3 + 2];
        int best_i = 0;
        Dtype best = 0;
        int end_ka = end_k - (end_k & 3);
        if (end_ka == CAFFE_CUDA_NUM_THREADS) {
          for (int k = 0; k < CAFFE_CUDA_NUM_THREADS; k += 4) {
            {
              Dtype x2 = buf[k * 3 + 0] - x1;
              Dtype y2 = buf[k * 3 + 1] - y1;
              Dtype z2 = buf[k * 3 + 2] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best   = d;
                best_i = k + k2;
              }
            }
            {
              Dtype x2 = buf[k * 3 + 3] - x1;
              Dtype y2 = buf[k * 3 + 4] - y1;
              Dtype z2 = buf[k * 3 + 5] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best   = d;
                best_i = k + k2 + 1;
              }
            }
            {
              Dtype x2 = buf[k * 3 + 6] - x1;
              Dtype y2 = buf[k * 3 + 7] - y1;
              Dtype z2 = buf[k * 3 + 8] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best   = d;
                best_i = k + k2 + 2;
              }
            }
            {
              Dtype x2 = buf[k * 3 + 9] - x1;
              Dtype y2 = buf[k * 3 + 10] - y1;
              Dtype z2 = buf[k * 3 + 11] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best   = d;
                best_i = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              Dtype x2 = buf[k * 3 + 0] - x1;
              Dtype y2 = buf[k * 3 + 1] - y1;
              Dtype z2 = buf[k * 3 + 2] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || d < best) {
                best   = d;
                best_i = k + k2;
              }
            }
            {
              Dtype x2 = buf[k * 3 + 3] - x1;
              Dtype y2 = buf[k * 3 + 4] - y1;
              Dtype z2 = buf[k * 3 + 5] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best   = d;
                best_i = k + k2 + 1;
              }
            }
            {
              Dtype x2 = buf[k * 3 + 6] - x1;
              Dtype y2 = buf[k * 3 + 7] - y1;
              Dtype z2 = buf[k * 3 + 8] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best   = d;
                best_i = k + k2 + 2;
              }
            }
            {
              Dtype x2 = buf[k * 3 + 9] - x1;
              Dtype y2 = buf[k * 3 + 10] - y1;
              Dtype z2 = buf[k * 3 + 11] - z1;
              Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
              if (d < best) {
                best   = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          Dtype x2 = buf[k * 3 + 0] - x1;
          Dtype y2 = buf[k * 3 + 1] - y1;
          Dtype z2 = buf[k * 3 + 2] - z1;
          Dtype d  = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || d < best) {
            best   = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || diff[i * n_points1 + j] > best) {
          diff[i * n_points1 + j]    = best;
          indexes[i * n_points1 + j] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

template <typename Dtype>
void ChamferDistanceLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* m_dist1 = dist1_.mutable_gpu_data();
  Dtype* m_dist2 = dist2_.mutable_gpu_data();
  int* indexes1  = indexes1_.mutable_gpu_data();
  int* indexes2  = indexes2_.mutable_gpu_data();

  const int num       = bottom[0]->num();
  const int n_points1 = bottom[0]->channels();
  const int n_points2 = bottom[1]->channels();

  NmDistanceKernel<Dtype><<<dim3(32, 16, 1), CAFFE_CUDA_NUM_THREADS>>>(
    num, n_points1, bottom[0]->gpu_data(), n_points2, bottom[1]->gpu_data(),
    m_dist1, indexes1);
  NmDistanceKernel<Dtype><<<dim3(32, 16, 1), CAFFE_CUDA_NUM_THREADS>>>(
    num, n_points2, bottom[1]->gpu_data(), n_points1, bottom[0]->gpu_data(),
    m_dist2, indexes2);

  Dtype loss1(0), loss2(0);
  caffe_gpu_asum(dist1_.count(), dist1_.gpu_data(), &loss1);
  caffe_gpu_asum(dist2_.count(), dist2_.gpu_data(), &loss2);
  Dtype loss = (loss1 / n_points1 + loss2 / n_points2) / num;

  top[0]->mutable_cpu_data()[0] = loss;

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void NmDistanceGradKernel(int num,
                                     int n_points1,
                                     const Dtype* points1,
                                     int n_points2,
                                     const Dtype* points2,
                                     const Dtype grad_dist1,
                                     const int* idx1,
                                     Dtype* grad_points1,
                                     Dtype* grad_points2) {
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n_points1;
         j += blockDim.x * gridDim.y) {
      int k   = idx1[i * n_points1 + j];
      Dtype x1 = points1[(i * n_points1 + j) * 3 + 0];
      Dtype y1 = points1[(i * n_points1 + j) * 3 + 1];
      Dtype z1 = points1[(i * n_points1 + j) * 3 + 2];
      Dtype x2 = points2[(i * n_points2 + k) * 3 + 0];
      Dtype y2 = points2[(i * n_points2 + k) * 3 + 1];
      Dtype z2 = points2[(i * n_points2 + k) * 3 + 2];
      Dtype grad  = grad_dist1 * 2;
      atomicAdd(&(grad_points1[(i * n_points1 + j) * 3 + 0]), grad * (x1 - x2));
      atomicAdd(&(grad_points1[(i * n_points1 + j) * 3 + 1]), grad * (y1 - y2));
      atomicAdd(&(grad_points1[(i * n_points1 + j) * 3 + 2]), grad * (z1 - z2));
      atomicAdd(&(grad_points2[(i * n_points2 + k) * 3 + 0]), grad * (x2 - x1));
      atomicAdd(&(grad_points2[(i * n_points2 + k) * 3 + 1]), grad * (y2 - y1));
      atomicAdd(&(grad_points2[(i * n_points2 + k) * 3 + 2]), grad * (z2 - z1));
    }
  }
}

template <typename Dtype>
void ChamferDistanceLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  const int* indexes1 = indexes1_.gpu_data();
  const int* indexes2 = indexes2_.gpu_data();

  const int num       = bottom[0]->num();
  const int n_points1 = bottom[0]->channels();
  const int n_points2 = bottom[1]->channels();
  Dtype grad_dist1  = top[0]->cpu_diff()[0] / n_points1 / num;
  Dtype grad_dist2  = top[0]->cpu_diff()[0] / n_points2 / num;

  NmDistanceGradKernel<Dtype><<<dim3(1, 16, 1), CAFFE_CUDA_NUM_THREADS / 2>>>(
    num, n_points1, bottom[0]->gpu_data(), n_points2, bottom[1]->gpu_data(),
    grad_dist1, indexes1, bottom[0]->mutable_gpu_diff(),
    bottom[1]->mutable_gpu_diff());
  NmDistanceGradKernel<Dtype><<<dim3(1, 16, 1), CAFFE_CUDA_NUM_THREADS / 2>>>(
    num, n_points2, bottom[1]->gpu_data(), n_points1, bottom[0]->gpu_data(),
    grad_dist2, indexes2, bottom[1]->mutable_gpu_diff(),
    bottom[0]->mutable_gpu_diff());

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ChamferDistanceLossLayer);

}  // namespace caffe
