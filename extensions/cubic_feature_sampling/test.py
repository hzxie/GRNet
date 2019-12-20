# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-20 11:50:50
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-20 15:16:29
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in cubic_feature_sampling.cu

import os
import sys
import torch

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.cubic_feature_sampling import CubicFeatureSamplingFunction

ptcloud = torch.rand(1, 32, 3) * 2 - 1
cubic_features = torch.rand(1, 2, 8, 8, 8)
ptcloud.requires_grad = True
cubic_features.requires_grad = True
print(gradcheck(CubicFeatureSamplingFunction.apply, [ptcloud.double().cuda(), cubic_features.double().cuda()]))
