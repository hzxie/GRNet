# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-23 11:46:33
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-23 17:37:23
# @Email:  cshzxie@gmail.com

import logging
import numpy as np
import os
import torch

import utils.data_loaders
import utils.helpers
import utils.io

from time import time

from models.rgnet import RGNet
from models.refiner import Refiner

def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                   batch_size=1,
                                                   num_workers=cfg.CONST.NUM_WORKERS,
                                                   collate_fn=utils.data_loaders.collate_fn,
                                                   pin_memory=True,
                                                   shuffle=False)

    # Setup networks and initialize networks
    rgnet = RGNet(cfg)
    refiner = Refiner(cfg)

    if torch.cuda.is_available():
        rgnet = torch.nn.DataParallel(rgnet).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()

    # Load the pretrained model from a checkpoint
    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    rgnet.load_state_dict(checkpoint['rgnet'])
    refiner.load_state_dict(checkpoint['refiner'])

    # Switch models to evaluation mode
    rgnet.eval()
    refiner.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            sparse_ptcloud, global_features = rgnet(data)
            dense_ptcloud = refiner(sparse_ptcloud, global_features)
            output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', taxonomy_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            output_file_path = os.path.join(output_folder, '%s.h5' % model_id)
            ptcloud = dense_ptcloud.squeeze().cpu().numpy()
            choice = np.random.permutation(ptcloud.shape[0])
            ptcloud = ptcloud[choice[:2500]]
            utils.io.IO.put(output_file_path, ptcloud)

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s File = %s' % (model_idx, n_samples, taxonomy_id, model_id, output_file_path))
