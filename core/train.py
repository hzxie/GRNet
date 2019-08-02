# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import caffe
import logging
import os

import utils.data_loaders

from datetime import datetime

from models.rplnet import create_rplnet
from utils.solvers import get_solver


def train_net(cfg):
    train_dataset = utils.data_loaders.get_dataset(cfg.CONST.DATASET, cfg, 'train')
    val_dataset = utils.data_loaders.get_dataset(cfg.CONST.DATASET, cfg, 'val')
    train_data_transforms = []
    val_data_transforms = []
    train_data_layer = caffe.layers.Python(name='data',
                                           include={'phase': caffe.TRAIN},
                                           ntop=3,
                                           python_param={
                                               'module': 'utils.data_loaders',
                                               'layer': utils.data_loaders.get_data_layer(cfg.CONST.DATASET),
                                               'param_str': repr({
                                                   'cfg': cfg,
                                                   'subset': 'train',
                                                   'transforms': train_data_transforms
                                               })
                                           })
    val_data_layer = caffe.layers.Python(name='data',
                                         include={'phase': caffe.TEST},
                                         ntop=3,
                                         python_param={
                                             'module': 'utils.data_loaders',
                                             'layer': utils.data_loaders.get_data_layer(cfg.CONST.DATASET),
                                             'param_str': repr({
                                                 'cfg': cfg,
                                                 'subset': 'val',
                                                 'transforms': val_data_transforms
                                             })
                                         })

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create the networks
    train_net = create_rplnet(cfg, train_data_layer, 'train')
    val_net = create_rplnet(cfg, val_data_layer, 'val')

    # Set up the iters for solvers
    cfg.TEST.TEST_ITER = val_dataset.get_n_itrs()    # Test all samples during testing, batch size = 1
    cfg.TEST.TEST_FREQ_ITER = train_dataset.get_n_itrs()    # The value indicates n_itrs within an epoch
    cfg.TRAIN.SAVE_FREQ_ITER = cfg.TRAIN.SAVE_FREQ_EPOCH * cfg.TEST.TEST_FREQ_ITER
    cfg.TRAIN.STEP_SIZE_ITER = cfg.TRAIN.LR_MILESTONE_EPOCH * cfg.TEST.TEST_FREQ_ITER
    cfg.TRAIN.N_ITERS = cfg.TRAIN.N_EPOCHS * cfg.TEST.TEST_FREQ_ITER

    # Create the solvers
    solver = caffe.get_solver(get_solver(cfg, train_net, val_net))
    solver.solve()
