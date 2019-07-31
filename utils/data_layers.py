# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py

import caffe
import numpy as np

from enum import Enum, unique


class ShapeNetBatchLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def load_next_batch(self):
        return np.zeros((224, 224, 3)), np.zeros(16384, 3)


class ShapeNetDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'labels']
        param = eval(self.param_str)
        self.cfg = param['cfg']
        self.phase = param['phase']
        self.batch_loader = ShapeNetBatchLoader(self.cfg)

    def forward(self, bottom, top):
        """Load data."""
        for i in range(self.batch_size):
            # Use the batch loader to load the next data.
            img, ptcloud = self.batch_loader.load_next_batch()

            # Add directly to the caffe data layer
            top[0].data[i, ...] = img
            top[1].data[i, ...] = ptcloud

    def reshape(self, bottom, top):
        """There is no need to reshape the data, since the input is of fixed size (rows and columns)"""
        pass

    def backward(self, top, propagate_down, bottom):
        """These layers does not back propagate"""
        pass
