# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import caffe
import os


def create_rplnet(cfg, data_layer, phase='train', data_transforms=None):
    NET_PROTOTXT_FILE_PATH = os.path.join('prototxt', 'rplnet_%s.prototxt' % phase)

    net = caffe.NetSpec()
    net.rgb, net.depth, net.ptcloud = data_layer
    

    net_prototxt = net.to_proto()
    with open(NET_PROTOTXT_FILE_PATH, 'w') as f:
        f.write(str(net_prototxt))

    return NET_PROTOTXT_FILE_PATH
