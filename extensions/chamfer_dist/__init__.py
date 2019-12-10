# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:07:32
# @Email:  cshzxie@gmail.com

import sys
import torch
import numpy as np

import chamfer


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistance(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super(ChamferDistance, self).__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        return ChamferFunction.apply(xyz1, xyz2)
