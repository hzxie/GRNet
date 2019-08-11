# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(name='permutohedral',
      version='1.0.0',
      ext_modules=[
          CppExtension('permutohedral', [
              'permutohedral.cpp',
          ]),
          CUDAExtension('permutohedral', [
              'permutohedral_cuda.cpp',
              'permutohedral.cu',
              'math_utils.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
