# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import h5py
import logging
import numpy as np
import pyexr
import os
import sys

# References: http://confluence.sensetime.com/pages/viewpage.action?pageId=44650315
from config import cfg
try:
    sys.path.append(cfg.MEMCACHED.LIBRARY_PATH)
    import mc
except:
    pass


class IO:
    @classmethod
    def get(cls, mc_client, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.png', '.jpg']:
            return cls._read_img(mc_client, file_path)
        elif file_extension in ['.npy']:
            return cls._read_npy(mc_client, file_path)
        elif file_extension in ['.exr']:
            return cls._read_exr(mc_client, file_path)
        elif file_extension in ['.h5', '.hdf5']:
            return cls._read_hdf5(mc_client, file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_img(cls, mc_client, file_path):
        if mc_client is None:
            return cv2.imread(file_path, cv2.IMREAD_UNCHANGED) / 255.
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            img_array = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            return img / 255.

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, mc_client, file_path):
        if mc_client is None:
            return np.load(file_path)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            buf_bytes = buf.tobytes()
            if not buf_bytes[:6] == b'\x93NUMPY':
                raise Exception('Invalid npy file format.')

            header_size = int.from_bytes(buf_bytes[8:10], byteorder='little')
            header = eval(buf_bytes[10:header_size + 10])
            dtype = np.dtype(header['descr'])
            nd_array = np.frombuffer(buf[header_size + 10:], dtype).reshape(header['shape'])

            return nd_array

    @classmethod
    def _read_exr(cls, mc_client, file_path):
        return 1.0 / pyexr.open(file_path).get("Depth.Z").astype(np.float32)

    @classmethod
    def _read_hdf5(cls, mc_client, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]
