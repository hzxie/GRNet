import sys
import numpy as np
from config import *
from create_solver import create_solver_proto, create_solver
from predict import predict


def train(base_lr, train_prototxt, snapshot_prefix, init_model=None):

    lr = float(base_lr)
    test_iter = 10
    iter_size = 1
    test_interval = 50
    num_iter = 5000
    snapshot_iter = 50
    debug_info = False

    solver_proto = create_solver_proto(train_prototxt,
                                       train_prototxt,
                                       lr,
                                       snapshot_prefix,
                                       test_iter=test_iter,
                                       test_interval=test_interval,
                                       max_iter=num_iter,
                                       iter_size=iter_size,
                                       snapshot=snapshot_iter,
                                       debug_info=debug_info)
    solver = create_solver(solver_proto)

    if init_model is not None:
        solver.net.copy_from(init_model)

    solver.solve()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: ' + sys.argv[0] + ' <base_lr> <train_prototxt> \
<snapshot_prefix> <init_caffemodel(optional)>')
    elif len(sys.argv) < 5:
        train(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
