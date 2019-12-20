# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:38:01
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-20 12:44:41
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in chamfer.cu

import os
import sys
import torch

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.chamfer_dist import ChamferFunction

x = torch.rand(4, 64, 3).double()
y = torch.rand(4, 128, 3).double()
x.requires_grad = True
y.requires_grad = True
print(gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()]))
