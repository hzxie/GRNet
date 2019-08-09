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
__C.DATASETS.SHAPENET.N_RENDERINGS             = 24
__C.DATASETS.SHAPENET.N_POINTS                 = 16384
__C.DATASETS.SHAPENET.DEPTH_MAX_VALUE          = 255
__C.DATASETS.SHAPENET.K                        = []
__C.DATASETS.SHAPENET.RGB_IMG_PATH             = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/render_%02d_l.png'
__C.DATASETS.SHAPENET.DEPTH_IMG_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/depth_%02d_l.exr'
__C.DATASETS.SHAPENET.POINTS_PATH              = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetPoints/%s/%s.npy'


#
# Constants
#
__C.CONST                                      = edict()
__C.CONST.DEVICE                               = '0'
__C.CONST.NUM_WORKERS                          = 4
__C.CONST.DATASET                              = 'ShapeNet'
__C.CONST.DATASET_MEAN                         = [0.5, 0.5, 0.5]
__C.CONST.DATASET_STD                          = [0.5, 0.5, 0.5]
__C.CONST.IMG_W                                = 256
__C.CONST.IMG_H                                = 192
__C.CONST.CROP_IMG_W                           = 210
__C.CONST.CROP_IMG_H                           = 210
__C.CONST.N_POINTS                             = 1024

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
# Train
#
__C.TRAIN                                      = edict()
__C.TRAIN.BATCH_SIZE                           = 16
__C.TRAIN.N_EPOCHS                             = 250
__C.TRAIN.SAVE_FREQ                            = 25
__C.TRAIN.LEARNING_RATE                        = 1e-4
__C.TRAIN.LR_MILESTONES                        = [150]
__C.TRAIN.GAMMA                                = .5
__C.TRAIN.BETAS                                = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                         = 0
__C.TRAIN.RANDOM_BG_COLOR                      = [[225, 255], [225, 255], [225, 255]]

#
# Test
#
__C.TEST                                       = edict()
__C.TEST.RANDOM_BG_COLOR                       = [[240, 240], [240, 240], [240, 240]]
__C.TEST.METRIC_NAME                           = 'F-Score'
