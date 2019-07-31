# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging

from easydict import EasyDict as edict

__C                                            = edict()
cfg                                            = __C

#
# Dataset Config
#
__C.DATASETS                                   = edict()
__C.DATASETS.SHAPENET                          = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH       = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNetStereoRendering/%s/%s/render_%02d_l.png'
__C.DATASETS.SHAPENET.DEPTH_IMG_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNetStereoRendering/%s/%s/depth_%02d_l.png'
__C.DATASETS.SHAPENET.POINTS_PATH              = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNetPoints/%s/%s.npy'

#
# Constants
#
__C.CONST                                      = edict()
__C.CONST.DEVICE                               = '0'                       # For multi-gpus: 0, 1, 2, 3
__C.CONST.N_WORKERS                            = 4
__C.CONST.DATASET                              = 'ShapeNet'
__C.CONST.DATASET_MEAN                         = [0.5, 0.5, 0.5]
__C.CONST.DATASET_STD                          = [0.5, 0.5, 0.5]
__C.CONST.RANDOM_BG_COLOR                      = [[225, 255], [225, 255], [225, 255]]
__C.CONST.N_POINTS                             = 1024
__C.CONST.WEIGHTS                              = None

#
# Directories
#
__C.DIR                                        = edict()
__C.DIR.OUT_PATH                               = './output'

#
# Memcached
#
__C.MEMCACHED                                  = edict()
__C.MEMCACHED.ENABLED                          = False
__C.MEMCACHED.LIBRARY_PATH                     = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                    = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                    = '/mnt/lustre/share/memcached_client/client.conf'

#
# PAVI
#
__C.PAVI                                       = edict()
__C.PAVI.ENABLED                               = False
__C.PAVI.PROJECT_NAME                          = 'RPLNet'
__C.PAVI.MODEL_NAME                            = ''
__C.PAVI.TAGS                                  = []

#
# Train
#
__C.TRAIN                                      = edict()
__C.TRAIN.BATCH_SIZE                           = 16
__C.TRAIN.N_EPOCHS                             = 250
__C.TRAIN.POLICY                               = 'adam'     # available options: sgd, adam
__C.TRAIN.LEARNING_RATE                        = 1e-4
__C.TRAIN.LR_MILESTONES                        = [150]
__C.TRAIN.BETAS                                = (.9, .999)
__C.TRAIN.MOMENTUM                             = .9
__C.TRAIN.GAMMA                                = .5
__C.TRAIN.WEIGHT_DECAY                         = 5e-4
__C.TRAIN.SAVE_FREQ                            = 25

#
# Test
#
__C.TEST                                       = edict()
