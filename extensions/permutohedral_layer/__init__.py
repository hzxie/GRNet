# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-30 10:01:53
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-04 00:04:05
# @Email:  cshzxie@gmail.com

import sys
import torch
import numpy as np

import permutohedral


class PermutohedralFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, neighborhood_size, group, out_channels, do_skip_blur, weights, bias, bias_multiplier, data,
                features):
        outputs = permutohedral.forward(
            neighborhood_size,
            group,
            out_channels,
            do_skip_blur,
            data,
            features,
            features,
            weights,
            bias,
            bias_multiplier,
        )
        output = outputs[0]

        weights_size = torch.from_numpy(np.array(weights.size()))
        bias_size = torch.from_numpy(np.array(bias.size()))
        data_size = torch.from_numpy(np.array(data.size()))
        outputs = [weights_size, bias_size, data_size, bias_multiplier] + outputs[1:]
        ctx.save_for_backward(*outputs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved_tensors = ctx.saved_tensors
        weights_size = torch.Size(saved_tensors[0])
        bias_size = torch.Size(saved_tensors[1])
        data_size = torch.Size(saved_tensors[2])
        bias_multiplier = saved_tensors[3]
        saved_tensors = saved_tensors[4:]

        grad_weights = torch.zeros(weights_size).cuda()
        grad_bias = torch.zeros(bias_size).cuda()
        grad_data = torch.zeros(data_size).cuda()
        grad_weights, grad_bias, grad_data = permutohedral.backward(bias_multiplier.contiguous(),
                                                                    grad_output.contiguous(),
                                                                    grad_weights.contiguous(), grad_bias.contiguous(),
                                                                    grad_data.contiguous(), saved_tensors)

        return None, None, None, None, grad_weights, grad_bias, None, grad_data, None


class PermutohedralLayer(torch.nn.Module):
    def __init__(self,
                 in_channels_data,
                 in_channels_feature,
                 out_height,
                 out_width,
                 out_channels,
                 neighborhood_size=1,
                 group=1,
                 bias=True,
                 skip_blur=False):
        super(Permutohedral, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.skip_blur = skip_blur
        self.group = group
        self.out_channels = out_channels
        self.weights = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels_data // group, 1,
                         self.get_filter_size(neighborhood_size, in_channels_feature)))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, 1, out_channels)) if bias else torch.Tensor()
        self.bias_multiplier = torch.nn.Parameter(torch.ones(1, 1, 1, out_height *
                                                             out_width)) if bias else torch.Tensor()
        self.reset_parameters()

    def get_filter_size(self, neighborhood_size, in_channels):
        return (neighborhood_size + 1)**(in_channels + 1) - neighborhood_size**(in_channels + 1)

    def reset_parameters(self):
        self.weights.data.normal_(std=0.001)

    def forward(self, data, features):
        return PermutohedralFunction.apply(self.neighborhood_size, self.group, self.out_channels, self.skip_blur, data,
                                           features, features, self.bias, self.bias_multiplier)
