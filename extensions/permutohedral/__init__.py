# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import sys
import torch
import numpy as np

import permutohedral


class PermutohedralFunction(torch.autograd.Function):
    pass


class Permutohedral(torch.nn.Module):
    def __init__(self):
        super(Permutohedral, self).__init__()

    def forward(self):
        return PermutohedralFunction.apply()
