# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-06 11:44:18
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from time import time
from tensorboardX import SummaryWriter

from extensions.chamfer_dist import ChamferDistance
from models.psgn import PSGN
from utils.average_meter import AverageMeter
from utils.metrics import Metrics


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, network=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.CONST.DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if network is None:
        network = RPLNet(cfg)

        if torch.cuda.is_available():
            network = torch.nn.DataParallel(network).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        network.load_state_dict(checkpoint['network_state_dict'])

    # Switch models to evaluation mode
    network.eval()

    # Set up loss functions
    loss = ChamferDistance()

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = AverageMeter()
    test_metrics = AverageMeter(Metrics.items())
    category_metrics = dict()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            ptcloud = network(data)
            dist1, dist2 = loss(ptcloud, data['gtcloud'])
            _loss = torch.mean(dist1) + torch.mean(dist2)
            test_losses.update(_loss.item() * 1000)
            _metrics = Metrics.get(ptcloud, data['gtcloud'])
            test_metrics.update(_metrics)

            if not taxonomy_id in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.items())
            category_metrics[taxonomy_id].update(_metrics)

            if test_writer is not None and model_idx < 3:
                pred_ptcloud = ptcloud.squeeze().cpu().numpy()
                gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
                pred_ptcloud_img = utils.helpers.get_ptcloud_img(pred_ptcloud)
                test_writer.add_image('Model%02d/Reconstructed' % model_idx, pred_ptcloud_img, epoch_idx + 1)
                gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud)
                test_writer.add_image('Model%02d/GroundTruth' % model_idx, gt_ptcloud_img, epoch_idx + 1)

            logging.info(
                'Test[%d/%d] Taxonomy = %s Sample = %s Loss = %.4f Metrics = %s' %
                (model_idx + 1, n_samples, taxonomy_id, model_id, test_losses.val(), ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch', test_losses.avg(), epoch_idx + 1)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx + 1)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
