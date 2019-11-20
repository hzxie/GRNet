# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-13 10:51:33
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-11-20 21:06:03
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='gridding',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('gridding', [
              'gridding_cuda.cpp',
              'gridding.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
