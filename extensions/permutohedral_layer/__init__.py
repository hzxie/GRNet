# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import sys
import torch
import numpy as np

import permutohedral


class PermutohedralFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, neighborhood_size, group, out_channels, do_skip_blur, use_bias_term, weights, bias,
                bias_multiplier, data, features):
        print('data', data.shape, data)
        print('features', features.shape, features)
        print('weights', weights.shape, weights)

        outputs = permutohedral.forward(
            neighborhood_size,
            group,
            out_channels,
            do_skip_blur,
            use_bias_term,
            data,
            features,
            features,
            weights,
            bias,
            bias_multiplier,
        )

        # ctx.save_for_backward(outputs['barycentric'], outputs['blur_neighbors'], outputs['max_idx'],
        #                       outputs['norm_back'], outputs['norm_there'], outputs['offset'])
        return outputs['output'][0]

    @staticmethod
    def backward(ctx, grad_data, grad_features):
        pass


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
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, 1, out_channels))
        self.bias_multiplier = torch.nn.Parameter(torch.ones(1, 1, 1, out_height * out_width))
        self.reset_parameters()

    def get_filter_size(self, neighborhood_size, in_channels):
        return (neighborhood_size + 1)**(in_channels + 1) - neighborhood_size**(in_channels + 1)

    def reset_parameters(self):
        self.weights.data.normal_(std=0.001)

    def forward(self, data, features):
        return PermutohedralFunction.apply(self.neighborhood_size, self.group, self.out_channels, self.skip_blur,
                                           self.bias is not None, data, features, features, self.bias,
                                           self.bias_multiplier)
