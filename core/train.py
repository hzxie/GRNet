# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-16 11:03:44
# @Email:  cshzxie@gmail.com

import logging
import os
import torch

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from datetime import datetime
from time import time
from tensorboardX import SummaryWriter

from core.test import test_net
from extensions.chamfer_dist import ChamferDistance
from extensions.gridding import Gridding, GriddingLoss
from models.rgnet import RGNet
from utils.average_meter import AverageMeter
from utils.metrics import Metrics

import matplotlib.pyplot as plt


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.VAL),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

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
    network = RGNet(cfg)
    network.apply(utils.helpers.init_weights)
    logging.debug('Parameters in network: %d.' % utils.helpers.count_parameters(network))

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=cfg.TRAIN.LR_MILESTONES,
                                                        gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(scales=cfg.NETWORK.GRIDDING_LOSS_SCALES, alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_index']
        best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        network.load_state_dict(checkpoint['network'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' %
                     (init_epoch, best_metrics))

    # Training/Testing the network
    losses = AverageMeter()
    for epoch_idx in range(init_epoch, cfg.TRAIN.N_EPOCHS):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        network.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            ptcloud = network(data)
            dist1, dist2 = chamfer_dist(ptcloud, data['gtcloud'])
            closs = torch.mean(dist1) + torch.mean(dist2)
            gloss = gridding_loss(ptcloud, data['gtcloud'])
            _loss = closs + gloss
            losses.update(_loss.item() * 1000)

            network.zero_grad()
            _loss.backward()
            optimizer.step()

            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('Loss/Batch', losses.val(), n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f' %
                         (epoch_idx + 1, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(),
                          data_time.val(), losses.val()))

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch', losses.avg(), epoch_idx + 1)
        logging.info('Epoch [%d/%d] EpochTime = %.3f (s) Loss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, losses.avg()))

        # Validate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, network)

        # Save ckeckpoints
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % (epoch_idx + 1)
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx + 1,
                'best_metrics': metrics.state_dict(),
                'network': network.state_dict()
            }, output_path) # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics

    train_writer.close()
    val_writer.close()
