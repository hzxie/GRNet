# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import logging
import numpy as np
import random
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
    def __init__(self, cfg, file_list, transforms=None, mc_client=None):
        self.cfg = cfg
        self.file_list = file_list
        self.transforms = transforms
        self.mc_client = mc_client

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.cfg['n_renderings'] - 1) if self.cfg['shuffle'] else 0
        for ri in self.cfg['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]

            data[ri] = IO.get(self.mc_client, file_path)

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

    def get_dataset(self, subset, transforms=None):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        return Dataset(
            {
                'n_renderings': self.cfg.DATASETS.SHAPENET.N_RENDERINGS,
                'required_items': ['rgb_img', 'depth_img', 'ptcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms, self.mc_client)

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
                    'ptcloud_path':
                    cfg.DATASETS.SHAPENET.POINTS_PATH % (dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader
} # yapf: disable
