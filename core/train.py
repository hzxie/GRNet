# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import caffe

import utils.data_layers

from models.rplnet import create_rplnet
from utils.solvers import get_solver


def train_net(cfg):
    train_data_layer = caffe.layers.Python(name='data',
                                           include={'phase': caffe.TRAIN},
                                           ntop=2,
                                           python_param={
                                               'module': 'utils.data_layers',
                                               'layer': 'ShapeNetDataLayer',
                                               'param_str': repr({
                                                   'cfg': cfg,
                                                   'phase': 'train'
                                               })
                                           })
    val_data_layer = caffe.layers.Python(name='data',
                                         include={'phase': caffe.TEST},
                                         ntop=2,
                                         python_param={
                                             'module': 'utils.data_layers',
                                             'layer': 'ShapeNetDataLayer',
                                             'param_str': repr({
                                                 'cfg': cfg,
                                                 'phase': 'val'
                                             })
                                         })

    train_net = create_rplnet(cfg, train_data_layer, 'train')
    val_net = create_rplnet(cfg, val_data_layer, 'val')
    solver = caffe.get_solver(get_solver(cfg, train_net, val_net))
