# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py

import caffe
import json
import logging
import random
import numpy as np

import utils.data_transforms

from easydict import EasyDict as edict
from enum import Enum, unique
from tqdm import tqdm

from utils.io import IO
# References: http://confluence.sensetime.com/pages/viewpage.action?pageId=44650315
from config import cfg
try:
    sys.path.append(cfg.MEMCACHED.LIBRARY_PATH)
    import mc
except:
    pass


def get_dataset(dataset_name, cfg, subset):
    DATASETS = {
        'ShapeNet': ShapeNetDataset,
    }

    return DATASETS[dataset_name](cfg, subset)


def get_data_layer(dataset_name):
    return '%sDataLayer' % dataset_name


class BatchLoader(object):
    def __init__(self, cfg, file_list, subset, transforms=None, mc_client=None):
        self.cfg = cfg
        self.file_list = file_list
        self.shuffle = True if subset == 'train' else False
        self.transformers = utils.data_transforms.Compose(transforms)
        self.mc_client = mc_client
        self.index = 0
        self.count = len(self.file_list)
        self._reset()

    def _reset(self):
        self.index = 0

        if self.shuffle:
            self.file_list = random.shuffle(self.file_list)

    def next(self):
        if self.index >= self.count:
            self._reset()

        file = self.file_list[self.index]
        raw_data = {}
        rand_idx = random.randint(0, cfg['n_renderings'] - 1) if self.shuffle else 0
        for ri in self.cfg['required_items']:
            file_path = file['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]

            raw_data[ri] = IO.get(self.mc_client, file_path)

        if self.transformers is not None:
            data = self.transformers(raw_data)

        self.index += 1
        return data


class ShapeNetDataset(object):
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self.subset = subset
        self.file_list = self._get_file_list(cfg, subset)

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []
        with open(self.cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            data_cateogries = json.loads(f.read())

        for dc in data_cateogries:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]
            for s in tqdm(samples, leave=False):
                file_list.append({
                    'rgb_img_path': [
                        cfg.DATASETS.SHAPENET.RGB_IMG_PATH % (dc['taxonomy_id'], s, i)
                        for i in range(cfg.DATASETS.SHAPENET.N_RENDERINGS)
                    ],
                    'depth_img_path': [
                        cfg.DATASETS.SHAPENET.DEPTH_IMG_PATH % (dc['taxonomy_id'], s, i)
                        for i in range(cfg.DATASETS.SHAPENET.N_RENDERINGS)
                    ],
                    'ptcloud_path':
                    cfg.DATASETS.SHAPENET.POINTS_PATH % (dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list

    def get_n_itrs(self):
        batch_size = self.cfg.TRAIN.BATCH_SIZE if self.subset == 'train' else 1
        return len(self.file_list) // batch_size


class ShapeNetDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['rgb', 'depth', 'ptcloud']

        # Parse Python parameters
        param = eval(self.param_str)
        self.cfg = edict(param['cfg'])
        self.subset = param['subset']
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE if self.subset == 'train' else 1

        # Get file list
        self.dataset = ShapeNetDataset(self.cfg, self.subset)
        self.file_list = self.dataset.file_list

        # Set up MemCached if available
        mc_client = None
        if self.cfg.MEMCACHED.ENABLED:
            self.mc_client = mc.MemcachedClient.GetInstance(cfg.MEMCACHED.SERVER_CONFIG, cfg.MEMCACHED.CLIENT_CONFIG)

        # Set up batch loader
        cfg = {
            'n_renderings': self.cfg.DATASETS.SHAPENET.N_RENDERINGS,
            'required_items': ['rgb_img', 'depth_img', 'ptcloud']
        }
        self.batch_loader = BatchLoader(cfg, self.file_list, self.subset, param['transforms'], mc_client)

        # Reshape Tops
        top[0].reshape(self.batch_size, 3, self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W)
        top[1].reshape(self.batch_size, 1, self.cfg.CONST.IMG_H, self.cfg.CONST.IMG_W)
        top[2].reshape(self.batch_size, 3, self.cfg.DATASETS.SHAPENET.N_POINTS)

    def forward(self, bottom, top):
        """Load data."""
        for i in range(self.batch_size):
            # Use the batch loader to load the next data.
            data = self.batch_loader.next()

            # Add directly to the caffe data layer
            top[0].data[i, ...] = data['rgb_img']
            top[1].data[i, ...] = data['depth_img']
            top[2].data[i, ...] = data['ptcloud']

    def reshape(self, bottom, top):
        """There is no need to reshape the data, since the input is of fixed size (rows and columns)"""
        pass

    def backward(self, top, propagate_down, bottom):
        """These layers does not back propagate"""
        pass
