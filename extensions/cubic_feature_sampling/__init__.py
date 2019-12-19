# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-19 16:55:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-19 21:23:30
# @Email:  cshzxie@gmail.com

import torch

import cubic_feature_sampling


class CubicFeatureSamplingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale, ptcloud, cubic_features):
        point_features, grid_pt_indexes = cubic_feature_sampling.forward(scale, ptcloud, cubic_features)
        ctx.save_for_backward(torch.Tensor([scale]), grid_pt_indexes)
        return point_features

    @staticmethod
    def backward(ctx, grad_point_features):
        scale, grid_pt_indexes = ctx.saved_tensors
        scale = int(scale.item())
        grad_ptcloud, grad_cubic_features = cubic_feature_sampling.backward(scale, grad_point_features,
                                                                            grid_pt_indexes)
        return None, grad_ptcloud, grad_cubic_features


class CubicFeatureSampling(torch.nn.Module):
    def __init__(self):
        super(CubicFeatureSampling, self).__init__()

    def forward(self, scale, ptcloud, cubic_features):
        h_scale = scale / 2
        ptcloud = ptcloud * h_scale + h_scale
        return CubicFeatureSamplingFunction.apply(scale, ptcloud, cubic_features)
