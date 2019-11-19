# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-13 10:51:33
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-11-18 16:17:40
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gridding',
    version='1.0.0',
    ext_modules=[
        CUDAExtension('gridding', [
            'gridding_cuda.cpp',
            'gridding.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
