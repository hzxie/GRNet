from config import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from predict import predict

cnn_deploy_file = 'cnn_deploy.prototxt'
bnn_deploy_file = 'bnn_deploy.prototxt'

step_size = 100
max_iter = 5000

iter_values = np.arange(step_size, max_iter, step_size)

cnn_iou_values = np.zeros(len(iter_values))
bnn_iou_values = np.zeros(len(iter_values))

ct = 0
for itr in iter_values:
    cnn_model = 'snapshot_models/cnn_train_iter_' + str(itr) + '.caffemodel'
    [acc, cnn_iou_values[ct]] = predict(cnn_deploy_file, cnn_model)

    bnn_model = 'snapshot_models/bnn_train_iter_' + str(itr) + '.caffemodel'
    [acc, bnn_iou_values[ct]] = predict(bnn_deploy_file, bnn_model)
    ct += 1

font = {'size': 22}
matplotlib.rc('font', **font)
plt.plot(iter_values, cnn_iou_values, 'r', iter_values, bnn_iou_values, 'g', linewidth=4.0)
plt.legend(['CNN', 'BNN'], loc='lower right')
plt.xlabel('Iterations', fontsize=24)
plt.ylabel('Test IoU', fontsize=24)
plt.show()
