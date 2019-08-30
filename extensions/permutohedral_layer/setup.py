# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

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
