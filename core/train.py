# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import caffe
import logging
import os

import utils.data_loaders

from datetime import datetime
from tensorboardX import SummaryWriter
from time import time

from models.rplnet import get_rplnet
from utils.solvers import get_solver
from utils.average_meter import AverageMeter


def train_net(cfg):
    train_dataset = utils.data_loaders.get_dataset(cfg.CONST.DATASET, cfg, 'train')
    val_dataset = utils.data_loaders.get_dataset(cfg.CONST.DATASET, cfg, 'val')
    train_data_transforms = [{
        'callback': 'RandomCrop',
        'parameters': {
            'img_size': (cfg.CONST.IMG_H, cfg.CONST.IMG_W),
            'crop_size': (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)
        },
        'objects': ['rgb_img', 'depth_img']
    }, {
        'callback': 'RandomFlip',
        'parameters': None,
        'objects': ['rgb_img', 'depth_img']
    }, {
        'callback': 'RandomPermuteRGB',
        'parameters': None,
        'objects': ['rgb_img']
    }, {
        'callback': 'RandomBackground',
        'parameters': {
            'bg_color': cfg.TRAIN.RANDOM_BG_COLOR
        },
        'objects': ['rgb_img']
    }, {
        'callback': 'Normalize',
        'parameters': {
            'mean': cfg.CONST.DATASET_MEAN,
            'std': cfg.CONST.DATASET_STD
        },
        'objects': ['rgb_img']
    }, {
        'callback': 'ToTensor',
        'parameters': None,
        'objects': ['rgb_img', 'depth_img', 'ptcloud']
    }]
    val_data_transforms = [{
        'callback': 'CenterCrop',
        'parameters': {
            'img_size': (cfg.CONST.IMG_H, cfg.CONST.IMG_W),
            'crop_size': (cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W)
        },
        'objects': ['rgb_img', 'depth_img']
    }, {
        'callback': 'RandomBackground',
        'parameters': {
            'bg_color': cfg.TEST.RANDOM_BG_COLOR
        },
        'objects': ['rgb_img']
    }, {
        'callback': 'Normalize',
        'parameters': {
            'mean': cfg.CONST.DATASET_MEAN,
            'std': cfg.CONST.DATASET_STD
        },
        'objects': ['rgb_img']
    }, {
        'callback': 'ToTensor',
        'parameters': None,
        'objects': ['rgb_img', 'depth_img', 'ptcloud']
    }]
    train_data_layer = caffe.layers.Python(name='data',
                                           include={'phase': caffe.TRAIN},
                                           ntop=3,
                                           python_param={
                                               'module':
                                               'utils.data_loaders',
                                               'layer':
                                               utils.data_loaders.get_data_layer(cfg.CONST.DATASET),
                                               'param_str':
                                               repr({
                                                   'cfg': cfg,
                                                   'subset': 'train',
                                                   'transforms': train_data_transforms
                                               })
                                           })
    val_data_layer = caffe.layers.Python(name='data',
                                         include={'phase': caffe.TEST},
                                         ntop=3,
                                         python_param={
                                             'module':
                                             'utils.data_loaders',
                                             'layer':
                                             utils.data_loaders.get_data_layer(cfg.CONST.DATASET),
                                             'param_str':
                                             repr({
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

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Create the networks
    train_net = get_rplnet(cfg, train_data_layer, 'train')
    val_net = get_rplnet(cfg, val_data_layer, 'val')

    # Set up the iters for solvers
    cfg.TEST.TEST_ITER = val_dataset.get_n_itrs()    # Test all samples during testing, batch size = 1
    cfg.TEST.TEST_FREQ_ITER = train_dataset.get_n_itrs()    # The value indicates n_itrs within an epoch
    cfg.TRAIN.SAVE_FREQ_ITER = cfg.TRAIN.SAVE_FREQ_EPOCH * cfg.TEST.TEST_FREQ_ITER
    cfg.TRAIN.STEP_SIZE_ITER = cfg.TRAIN.LR_MILESTONE_EPOCH * cfg.TEST.TEST_FREQ_ITER
    cfg.TRAIN.N_ITERS = cfg.TRAIN.N_EPOCHS * cfg.TEST.TEST_FREQ_ITER

    # Create the solvers
    solver = caffe.get_solver(get_solver(cfg, train_net, val_net))

    # Training/Testing the network
    losses = AverageMeter()
    for itr_idx in range(cfg.TRAIN.N_ITERS):
        _time = time()

        epoch_idx = itr_idx / cfg.TEST.TEST_FREQ_ITER
        batch_idx = itr_idx % cfg.TEST.TEST_FREQ_ITER

        solver.step(1)
        loss = solver.net.blobs['loss'].data
        losses.update(loss)
        train_writer.add_scalar('BatchLoss', loss, itr_idx + 1)

        if itr_idx % cfg.TEST.TEST_FREQ_ITER == 0:
            test_loss = solver.test_nets[0].blobs['loss'].data
            train_writer.add_scalar('EpochLoss', losses.avg, itr_idx + 1)
            val_writer.add_scalar('EpochLoss', test_loss, itr_idx + 1)
            losses.reset()

        logging.info('[Epoch %d/%d][Batch %d/%d] Time = %.3f(s) Loss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.N_EPOCHS, batch_idx + 1, cfg.TEST.TEST_FREQ_ITER, time() - _time, loss))

    train_writer.close()
    val_writer.close()