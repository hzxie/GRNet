# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os

from caffe.proto import caffe_pb2


def get_solver(cfg, train_net, test_net):
    SOLVER_PROTOTXT_FILE_PATH = os.path.join('prototxt', 'solver.prototxt')
    SOLVER_SNAPSHOT_PREFIX = os.path.join(cfg.DIR.CHECKPOINTS, 'solver_')

    solver = caffe_pb2.SolverParameter()
    solver.solver_type = caffe_pb2.SolverParameter.ADAM
    solver.momentum = cfg.TRAIN.BETAS[0]
    solver.momentum2 = cfg.TRAIN.BETAS[1]
    solver.weight_decay = cfg.TRAIN.WEIGHT_DECAY

    solver.base_lr = cfg.TRAIN.LEARNING_RATE
    solver.lr_policy = 'step'
    solver.stepsize = cfg.TRAIN.STEP_SIZE_ITER
    solver.gamma = cfg.TRAIN.GAMMA

    solver.max_iter = cfg.TRAIN.N_ITERS
    solver.snapshot = cfg.TRAIN.SAVE_FREQ_ITER
    solver.snapshot_prefix = SOLVER_SNAPSHOT_PREFIX
    solver.display = 1
    solver.iter_size = 1

    solver.train_net = train_net
    solver.test_net.extend([test_net])
    solver.test_iter.extend([cfg.TEST.TEST_ITER])
    solver.test_interval = cfg.TEST.TEST_FREQ_ITER

    with open(SOLVER_PROTOTXT_FILE_PATH, 'w') as f:
        f.write(str(solver))

    return SOLVER_PROTOTXT_FILE_PATH
