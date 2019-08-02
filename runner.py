#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import argparse
import caffe
import cv2
import logging
import matplotlib
import os
import sys
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')
# Fix deadlock in DataLoader
cv2.setNumThreads(0)

from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of R2Net runner')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to use', default=cfg.CONST.DEVICE, type=int)
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    caffe.set_mode_gpu()
    caffe.set_device(cfg.CONST.DEVICE)

    # Start train/test process
    if not args.test:
        train_net(cfg)
    else:
        if not 'WEIGHTS' in cfg.CONST or not os.path.exists(cfg.CONST.WEIGHTS):
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)

        test_net(cfg)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on https://github.com/hzxie/RPLNet")

    logging.basicConfig(
        format='%(levelname)s %(asctime)s.%(msecs)03d %(process)d %(filename)s:%(lineno)s] %(message)s',
        datefmt='%m%d %H:%M:%S',
        level=logging.DEBUG)
    main()
