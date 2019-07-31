# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from caffe.proto import caffe_pb2


def get_solver(cfg, train_net, test_net):
    SOLVER_PROTOTXT_FILE_PATH = 'prototxt/solver.prototxt'

    solver = caffe_pb2.SolverParameter()
    solver.train_net = train_net
    solver.test_net.extend([test_net])
    solver.test_iter.extend([1])
    solver.test_interval = 1

    with open(SOLVER_PROTOTXT_FILE_PATH, 'w') as f:
        f.write(str(solver))

    return SOLVER_PROTOTXT_FILE_PATH
