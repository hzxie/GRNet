# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 10:22:03
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 12:27:57
# @Email:  cshzxie@gmail.com

import cv2
import pcl
import logging
import numpy as np
import pyexr
import os
import sys

from io import BytesIO

# References: http://confluence.sensetime.com/pages/viewpage.action?pageId=44650315
from config import cfg
sys.path.append(cfg.MEMCACHED.LIBRARY_PATH)

mc_client = None
if cfg.MEMCACHED.ENABLED:
    import mc
    mc_client = mc.MemcachedClient.GetInstance(cfg.MEMCACHED.SERVER_CONFIG, cfg.MEMCACHED.CLIENT_CONFIG)


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.png', '.jpg']:
            return cls._read_img(file_path)
        elif file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.exr']:
            return cls._read_exr(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_img(cls, file_path):
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
    def _read_npy(cls, file_path):
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
    def _read_exr(cls, file_path):
        return 1.0 / pyexr.open(file_path).get("Depth.Z").astype(np.float32)

    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        if mc_client is None:
            pc = pcl.PointCloud()
            pc.from_file(file_path.encode())
            ptcloud = np.array(pc.to_list())
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            text = mc.ConvertString(pyvector).split('\n')
            start_line_idx = len(text) - 1
            for idx, line in enumerate(text):
                if line == 'DATA ascii':
                    start_line_idx = idx + 1
                    break

            ptcloud = text[start_line_idx:]
            ptcloud = np.genfromtxt(BytesIO('\n'.join(ptcloud).encode()), dtype=np.float32)

        ptcloud = np.concatenate((ptcloud, np.array([[0, 0, 0]])), axis=0)
        return ptcloud
