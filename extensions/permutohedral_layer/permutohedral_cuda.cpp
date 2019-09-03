/*
 * @Author: Haozhe Xie
 * @Date:   2019-08-30 10:01:53
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2019-09-03 17:56:27
 * @Email:  cshzxie@gmail.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "permutohedral_cuda.hpp"

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> permutohedral_cuda_forward(
  const cublasHandle_t& handle,
  int neighborhood_size,
  int group,
  int out_channels,
  bool do_skip_blur,
  torch::Tensor data,
  torch::Tensor in_features,
  torch::Tensor out_features,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor bias_multiplier);

std::vector<torch::Tensor> permutohedral_cuda_backward(
  const cublasHandle_t& handle,
  const torch::Tensor bias_multiplier,
  const torch::Tensor output_grad,
  torch::Tensor grad_weights,
  torch::Tensor grad_bias,
  torch::Tensor grad_data,
  std::vector<torch::Tensor> saved_tensors);

std::vector<torch::Tensor> permutohedral_forward(
  int neighborhood_size,
  int group,
  int out_channels,
  bool do_skip_blur,
  torch::Tensor data,
  torch::Tensor in_features,
  torch::Tensor out_features,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor bias_multiplier) {
  CHECK_INPUT(data);
  CHECK_INPUT(in_features);
  CHECK_INPUT(out_features);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(bias_multiplier);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  return permutohedral_cuda_forward(
    handle, neighborhood_size, group, out_channels, do_skip_blur, data,
    in_features, out_features, weights, bias, bias_multiplier);
}

std::vector<torch::Tensor> permutohedral_backward(
  const torch::Tensor bias_multiplier,
  const torch::Tensor grad_output,
  torch::Tensor grad_weights,
  torch::Tensor grad_bias,
  torch::Tensor grad_data,
  std::vector<torch::Tensor> saved_tensors) {
  const int N_TENSORS = 9;

  CHECK_INPUT(bias_multiplier);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(grad_weights);
  CHECK_INPUT(grad_bias);
  CHECK_INPUT(grad_data);
  for (size_t i = 0; i < saved_tensors.size(); ++i) {
    if (i % N_TENSORS == 2) {
      // Skip the tensor for constants
      continue;
    }
    CHECK_INPUT(saved_tensors.at(i));
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  return permutohedral_cuda_backward(handle, bias_multiplier, grad_output,
                                     grad_weights, grad_bias, grad_data,
                                     saved_tensors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &permutohedral_forward, "Permutohedral forward (CUDA)");
  m.def("backward", &permutohedral_backward, "Permutohedral backward (CUDA)");
}
