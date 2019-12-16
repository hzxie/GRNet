# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-16 11:05:25
# @Email:  cshzxie@gmail.com

import logging

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 16384
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/complete/%s/%s.pcd'
__C.DATASETS.SHAPENET_CAR                        = edict()
__C.DATASETS.SHAPENET_CAR.CATEGORY_FILE_PATH     = './datasets/ShapeNet-Car.json'
__C.DATASETS.SHAPENET_CAR.N_POINTS               = 16384
__C.DATASETS.SHAPENET_CAR.PARTIAL_POINTS_PATH    = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCarCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET_CAR.COMPLETE_POINTS_PATH   = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCarCompletion/%s/complete/%s/%s.pcd'
__C.DATASETS.KITTI                               = edict()
__C.DATASETS.KITTI.CATEGORY_FILE_PATH            = './datasets/KITTI.json'
__C.DATASETS.SHAPENET_RGBD                       = edict()
__C.DATASETS.SHAPENET_RGBD.CATEGORY_FILE_PATH    = './datasets/ShapeNet-RGBD.json'
__C.DATASETS.SHAPENET_RGBD.N_RENDERINGS          = 24
__C.DATASETS.SHAPENET_RGBD.N_POINTS              = 16384
__C.DATASETS.SHAPENET_RGBD.K                     = []
__C.DATASETS.SHAPENET_RGBD.RGB_IMG_PATH          = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/render_%02d_l.png'
__C.DATASETS.SHAPENET_RGBD.DEPTH_IMG_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetStereoRendering/%s/%s/depth_%02d_l.exr'
__C.DATASETS.SHAPENET_RGBD.POINTS_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetPoints/%s/%s.npy'

#
# Dataset
#
__C.DATASET                                      = edict()
__C.DATASET.MEAN                                 = [0.5, 0.5, 0.5]
__C.DATASET.STD                                  = [0.5, 0.5, 0.5]
# Dataset Options: ShapeNet, ShapeNetCars, ShapeNetRGBD, KITTI
__C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
__C.DATASET.TEST_DATASET                         = 'ShapeNet'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '0'
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.IMG_W                                  = 224
__C.CONST.IMG_H                                  = 224
__C.CONST.CROP_IMG_W                             = 210
__C.CONST.CROP_IMG_H                             = 210
__C.CONST.N_INPUT_POINTS                         = 2048

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.GRIDDING_LOSS_SCALES                 = [128, 64, 32, 16]
__C.NETWORK.GRIDDING_LOSS_ALPHAS                 = [200, 20, 5, 1]

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 32
__C.TRAIN.N_EPOCHS                               = 250
__C.TRAIN.SAVE_FREQ                              = 25
__C.TRAIN.LEARNING_RATE                          = 1e-4
__C.TRAIN.LR_MILESTONES                          = [150]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 5e-5
__C.TRAIN.RANDOM_BG_COLOR                        = [[225, 255], [225, 255], [225, 255]]

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.RANDOM_BG_COLOR                         = [[240, 240], [240, 240], [240, 240]]
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
