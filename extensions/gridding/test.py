# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:48:55
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-20 12:44:19
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in gridding.cu and gridding_reverse.cu

import os
import sys
import torch

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.gridding import GriddingFunction, GriddingReverseFunction

x = torch.rand(2, 4, 4, 4)
x.requires_grad = True
gradcheck(GriddingReverseFunction.apply, [4, x.double().cuda()])

x = torch.rand(4, 8, 8, 8)
x.requires_grad = True
gradcheck(GriddingReverseFunction.apply, [8, x.double().cuda()])

x = torch.rand(1, 16, 16, 16)
x.requires_grad = True
gradcheck(GriddingReverseFunction.apply, [16, x.double().cuda()])

y = torch.rand(1, 32, 3)
y.requires_grad = True
gradcheck(GriddingFunction.apply, [y.double().cuda()])
