# -*- coding: utf-8 -*-
#
# Developed by Thibault GROUEIX <thibault.groueix.2012@polytechnique.org>

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer',
    version='1.0.0',
    ext_modules=[
        CUDAExtension('chamfer', [
            'chamfer_cuda.cpp',
            'chamfer.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
