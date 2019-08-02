# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import caffe
import os


def get_rplnet(cfg, data_layer, phase='train', data_transforms=None):
    NET_PROTOTXT_FILE_PATH = os.path.join('prototxt', 'rplnet_%s.prototxt' % phase)

    net = caffe.NetSpec()
    net.rgb, net.depth, net.ptcloud = data_layer
    net.conv1 = caffe.layers.Convolution(net.rgb,
                                         kernel_size=3,
                                         num_output=32,
                                         pad=1,
                                         stride=1,
                                         weight_filler=dict(type='xavier'))
    net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
    net.conv2 = caffe.layers.Convolution(net.pool1,
                                         kernel_size=3,
                                         num_output=32,
                                         pad=1,
                                         stride=1,
                                         weight_filler=dict(type='xavier'))
    net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
    net.conv3 = caffe.layers.Convolution(net.pool2,
                                         kernel_size=3,
                                         num_output=32,
                                         pad=1,
                                         stride=1,
                                         weight_filler=dict(type='xavier'))
    net.pool3 = caffe.layers.Pooling(net.conv3, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
    net.conv4 = caffe.layers.Convolution(net.pool3,
                                         kernel_size=3,
                                         num_output=32,
                                         pad=1,
                                         stride=1,
                                         weight_filler=dict(type='xavier'))
    net.pool4 = caffe.layers.Pooling(net.conv4, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
    net.conv5 = caffe.layers.Convolution(net.pool4,
                                         kernel_size=3,
                                         num_output=1,
                                         pad=1,
                                         stride=1,
                                         weight_filler=dict(type='xavier'))
    net.pool5 = caffe.layers.Pooling(net.conv5, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
    net.conv6 = caffe.layers.Convolution(net.pool5,
                                         kernel_size=3,
                                         num_output=1,
                                         pad=1,
                                         stride=1,
                                         weight_filler=dict(type='xavier'))
    net.pool6 = caffe.layers.Pooling(net.conv6, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)

    net_prototxt = net.to_proto()
    with open(NET_PROTOTXT_FILE_PATH, 'w') as f:
        f.write(str(net_prototxt))

    return NET_PROTOTXT_FILE_PATH
