# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-11-06 17:09:17
# @Email:  cshzxie@gmail.com

import json
import logging
import numpy as np
import random
import sys
import torch.utils.data.dataset

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


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None, mc_client=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.mc_client = mc_client

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]

            data[ri] = IO.get(self.mc_client, file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # Set up MemCached if available
        self.mc_client = None
        if cfg.MEMCACHED.ENABLED:
            self.mc_client = mc.MemcachedClient.GetInstance(cfg.MEMCACHED.SERVER_CONFIG, cfg.MEMCACHED.CLIENT_CONFIG)

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms, self.mc_client)

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback': 'ToTensor',
            'parameters': None,
            'objects': ['partial_cloud', 'gtcloud']
        }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path': [
                        cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gtcloud_path': cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetRgbdDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # Set up MemCached if available
        self.mc_client = None
        if cfg.MEMCACHED.ENABLED:
            self.mc_client = mc.MemcachedClient.GetInstance(cfg.MEMCACHED.SERVER_CONFIG, cfg.MEMCACHED.CLIENT_CONFIG)

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset(
            {
                'n_renderings': self.cfg.DATASETS.SHAPENET.N_RENDERINGS,
                'required_items': ['rgb_img', 'depth_img', 'gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms, self.mc_client)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
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
                'callback': 'RandomBackground',
                'parameters': {
                    'bg_color': cfg.TRAIN.RANDOM_BG_COLOR
                },
                'objects': ['rgb_img']
            }, {
                'callback': 'RandomPermuteRGB',
                'parameters': None,
                'objects': ['rgb_img']
            }, {
                'callback': 'Normalize',
                'parameters': {
                    'mean': cfg.DATASET.MEAN,
                    'std': cfg.DATASET.STD
                },
                'objects': ['rgb_img']
            }, {
                'callback': 'ToTensor',
                'parameters': None,
                'objects': ['rgb_img', 'depth_img', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
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
                    'mean': cfg.DATASET.MEAN,
                    'std': cfg.DATASET.STD
                },
                'objects': ['rgb_img']
            }, {
                'callback': 'ToTensor',
                'parameters': None,
                'objects': ['rgb_img', 'depth_img', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]
            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'rgb_img_path': [
                        cfg.DATASETS.SHAPENET.RGB_IMG_PATH % (dc['taxonomy_id'], s, i)
                        for i in range(cfg.DATASETS.SHAPENET.N_RENDERINGS)
                    ],
                    'depth_img_path': [
                        cfg.DATASETS.SHAPENET.DEPTH_IMG_PATH % (dc['taxonomy_id'], s, i)
                        for i in range(cfg.DATASETS.SHAPENET.N_RENDERINGS)
                    ],
                    'gtcloud_path':
                    cfg.DATASETS.SHAPENET.POINTS_PATH % (dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetRGBD': ShapeNetRgbdDataLoader,
} # yapf: disable
