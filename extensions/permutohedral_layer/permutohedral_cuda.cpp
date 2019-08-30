// Copyright 2019 Haozhe Xie
// Distributed under the MIT Software license,
// (See https://opensource.org/licenses/MIT)
// 
// References:
// - https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda.cpp

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

std::map<std::string, std::vector<torch::Tensor> > permutohedral_cuda_forward(
  const cublasHandle_t& handle,
  int neighborhood_size,
  int group,
  bool do_skip_blur,
  bool use_bias_term,
  torch::Tensor data,
  torch::Tensor in_features,
  torch::Tensor out_features,
  torch::Tensor weights,
  torch::Tensor bias,
  torch::Tensor bias_multiplier);

std::vector<torch::Tensor> permutohedral_cuda_backward();

std::map<std::string, std::vector<torch::Tensor> > permutohedral_forward(
  int neighborhood_size,
  int group,
  bool do_skip_blur,
  bool use_bias_term,
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
  CHECK_INPUT(bias_multiplier)
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  return permutohedral_cuda_forward(
    handle, neighborhood_size, group, do_skip_blur, use_bias_term, data,
    in_features, out_features, weights, bias, bias_multiplier);
}

std::vector<torch::Tensor> permutohedral_backward() {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &permutohedral_forward, "Permutohedral forward (CUDA)");
  m.def("backward", &permutohedral_backward, "Permutohedral backward (CUDA)");
}
