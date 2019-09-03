# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-30 10:01:53
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-03 10:03:24
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='permutohedral',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('permutohedral', [
              'permutohedral_cuda.cpp',
              'permutohedral.cu',
              'math_utils.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
