# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

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
from models.psgn import PSGN
from utils.average_meter import AverageMeter
from utils.metrics import Metrics


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    train_transforms = utils.data_transforms.Compose([{
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
        'callback': 'RandomPermuteRGB',
        'parameters': None,
        'objects': ['rgb_img']
    }, {
        'callback': 'RandomBackground',
        'parameters': {
            'bg_color': cfg.TRAIN.RANDOM_BG_COLOR
        },
        'objects': ['rgb_img']
    }, {
        'callback': 'Normalize',
        'parameters': {
            'mean': cfg.CONST.DATASET_MEAN,
            'std': cfg.CONST.DATASET_STD
        },
        'objects': ['rgb_img']
    }, {
        'callback': 'ToTensor',
        'parameters': None,
        'objects': ['rgb_img', 'depth_img', 'ptcloud']
    }])
    val_transforms = utils.data_transforms.Compose([{
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
            'mean': cfg.CONST.DATASET_MEAN,
            'std': cfg.CONST.DATASET_STD
        },
        'objects': ['rgb_img']
    }, {
        'callback': 'ToTensor',
        'parameters': None,
        'objects': ['rgb_img', 'depth_img', 'ptcloud']
    }])

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.CONST.DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN, train_transforms),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.VAL, val_transforms),
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
    network = PSGN(cfg)
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
    loss = ChamferDistance()

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_epoch = checkpoint['best_epoch']
        best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        network.load_state_dict(checkpoint['network_state_dict'])
        logging.info('Recover complete. Current epoch #%d, best metrics = %s at epoch #%d.' %
                     (init_epoch, best_metrics, best_epoch))

    # Training/Testing the network
    losses = AverageMeter()
    for epoch_idx in range(init_epoch, cfg.TRAIN.N_EPOCHS):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        lr_scheduler.step()
        network.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            ptclouds = network(data)
            dist1, dist2 = loss(ptclouds, data['ptcloud'])
            _loss = torch.mean(dist1) + torch.mean(dist2)
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

        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch', losses.avg(), epoch_idx + 1)
        logging.info('Epoch [%d/%d] EpochTime = %.3f (s) Loss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, losses.avg()))

        # Validate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, network)

        # Save ckeckpoints
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'best-ckpt.pth' if metrics.better_than(best_metrics) else 'epoch-%03d.pth' % (epoch_idx + 1)
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx + 1,
                'best_metrics': metrics.state_dict(),
                'network': network.state_dict()
            }, output_path) # yapf: disable
            logging.info('Saved checkpoint to %s ...' % output_path)

    train_writer.close()
    val_writer.close()
