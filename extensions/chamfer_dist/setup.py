# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-03 10:01:38
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(name='chamfer',
      version='1.0.0',
      ext_modules=[
          CppExtension('chamfer', ['chamfer.cpp']),
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
