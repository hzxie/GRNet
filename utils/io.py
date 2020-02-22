# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 10:22:03
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:13:01
# @Email:  cshzxie@gmail.com

import cv2
import h5py
import numpy as np
import pyexr
import open3d
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
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def put(cls, file_path, file_content):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.pcd']:
            return cls._write_pcd(file_path, file_content)
        elif file_extension in ['.h5']:
            return cls._write_h5(file_path, file_content)
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
            pc = open3d.io.read_point_cloud(file_path)
            ptcloud = np.array(pc.points)
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

        # ptcloud = np.concatenate((ptcloud, np.array([[0, 0, 0]])), axis=0)
        return ptcloud

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        # Avoid overflow while gridding
        return f['data'][()] * 0.9

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _write_pcd(cls, file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc)

    @classmethod
    def _write_h5(cls, file_path, file_content):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=file_content)
