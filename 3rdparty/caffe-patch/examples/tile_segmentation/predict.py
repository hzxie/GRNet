import numpy as np
from config import *
from get_tile_data import get_tile_data
from compute_accuracy_iou import compute_accuracy_and_iou

[test_x, test_y] = get_tile_data(NUM_DATA['TEST'], RAND_SEED['TEST'])


def predict(prototxt, caffe_model):
    net = caffe.Net(prototxt, caffe_model, caffe.TEST)
    net.load_blobs_from(caffe_model)

    dinputs = {}
    dinputs['data'] = test_x

    predictions = net.forward_all(**dinputs)['conv_result']

    [accuracy, iou] = compute_accuracy_and_iou(predictions, test_y)

    print([accuracy, iou])

    return [accuracy, iou]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' <prototxt> <caffe_model>')
    else:
        predict(sys.argv[1], sys.argv[2])
