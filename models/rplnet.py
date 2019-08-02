# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import caffe
import os

from caffe import layers as L


def get_rplnet(cfg, data_layer, phase='train', data_transforms=None):
    NET_PROTOTXT_FILE_PATH = os.path.join('prototxt', 'rplnet_%s.prototxt' % phase)

    n = cfg.TRAIN.BATCH_SIZE if phase == 'train' else 1
    net = caffe.NetSpec()
    net.rgb, net.depth, net.gtcloud = data_layer

    net.rgbd = L.Concat(*[net.rgb, net.depth])
    net.conv1a = L.Convolution(net.rgbd, kernel_size=3, num_output=16, pad=1, weight_filler=dict(type='xavier'))
    net.relu1a = L.ReLU(net.conv1a, in_place=True)
    net.conv1b = L.Convolution(net.relu1a, kernel_size=3, num_output=16, pad=1, weight_filler=dict(type='xavier'))
    net.relu1b = L.ReLU(net.conv1b, in_place=True)
    net.conv2a = L.Convolution(net.relu1b,
                               kernel_size=3,
                               num_output=32,
                               pad=1,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.relu2a = L.ReLU(net.conv2a, in_place=True)
    net.conv2b = L.Convolution(net.relu2a, kernel_size=3, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    net.relu2b = L.ReLU(net.conv2b, in_place=True)
    net.conv2c = L.Convolution(net.relu2b, kernel_size=3, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    net.relu2c = L.ReLU(net.conv2c, in_place=True)
    net.conv3a = L.Convolution(net.relu2c,
                               kernel_size=3,
                               num_output=64,
                               pad=1,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.relu3a = L.ReLU(net.conv3a, in_place=True)
    net.conv3b = L.Convolution(net.relu3a, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.relu3b = L.ReLU(net.conv3b, in_place=True)
    net.conv3c = L.Convolution(net.relu3b, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.relu3c = L.ReLU(net.conv3c, in_place=True)
    net.conv4a = L.Convolution(net.relu3c,
                               kernel_size=3,
                               num_output=128,
                               pad=1,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.relu4a = L.ReLU(net.conv4a, in_place=True)
    net.conv4b = L.Convolution(net.relu4a, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.relu4b = L.ReLU(net.conv4b, in_place=True)
    net.conv4c = L.Convolution(net.relu4b, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.relu4c = L.ReLU(net.conv4c, in_place=True)
    net.conv5a = L.Convolution(net.relu4c,
                               kernel_size=5,
                               num_output=256,
                               pad=2,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.relu5a = L.ReLU(net.conv5a, in_place=True)
    net.conv5b = L.Convolution(net.relu5a, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.relu5b = L.ReLU(net.conv5b, in_place=True)
    net.conv5c = L.Convolution(net.relu5b, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.relu5c = L.ReLU(net.conv5c, in_place=True)
    net.conv6a = L.Convolution(net.relu5c,
                               kernel_size=5,
                               num_output=512,
                               pad=2,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.relu6a = L.ReLU(net.conv6a, in_place=True)
    net.conv6b = L.Convolution(net.relu6a, kernel_size=3, num_output=512, pad=1, weight_filler=dict(type='xavier'))
    net.relu6b = L.ReLU(net.conv6b, in_place=True)
    net.conv6c = L.Convolution(net.relu6b, kernel_size=3, num_output=512, pad=1, weight_filler=dict(type='xavier'))
    net.relu6c = L.ReLU(net.conv6c, in_place=True)
    net.conv6d = L.Convolution(net.relu6c, kernel_size=3, num_output=512, pad=1, weight_filler=dict(type='xavier'))
    net.relu6d = L.ReLU(net.conv6d, in_place=True)
    net.conv6e = L.Convolution(net.relu6d,
                               kernel_size=5,
                               num_output=512,
                               pad=2,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.relu6e = L.ReLU(net.conv6e, in_place=True)
    net.fc7a = L.InnerProduct(net.relu6e, num_output=2048, weight_filler=dict(type='gaussian', std=0.01))
    net.relu7a = L.ReLU(net.fc7a, in_place=True)
    net.fc7b = L.InnerProduct(net.relu7a, num_output=2048, weight_filler=dict(type='gaussian', std=0.01))

    net.deconv1a = L.Deconvolution(net.relu6e,
                                   convolution_param=dict(kernel_size=5,
                                                          num_output=256,
                                                          pad=2,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp1a = L.Interp(net.deconv1a, height=6, width=8)
    net.deconv1b = L.Convolution(net.relu6c, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.dsum1b = L.Eltwise(*[net.dinterp1a, net.deconv1b])
    net.drelu1b = L.ReLU(net.dsum1b, in_place=True)
    net.deconv1c = L.Convolution(net.drelu1b, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.drelu1c = L.ReLU(net.deconv1c, in_place=True)
    net.deconv2a = L.Deconvolution(net.drelu1c,
                                   convolution_param=dict(kernel_size=5,
                                                          num_output=128,
                                                          pad=2,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp2a = L.Interp(net.deconv2a, height=12, width=16)
    net.deconv2b = L.Convolution(net.relu5c, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.dsum2b = L.Eltwise(*[net.dinterp2a, net.deconv2b])
    net.drelu2b = L.ReLU(net.dsum2b, in_place=True)
    net.deconv2c = L.Convolution(net.drelu2b, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.drelu2c = L.ReLU(net.deconv2c, in_place=True)
    net.deconv3a = L.Deconvolution(net.drelu2c,
                                   convolution_param=dict(kernel_size=3,
                                                          num_output=64,
                                                          pad=1,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp3a = L.Interp(net.deconv3a, height=24, width=32)
    net.deconv3b = L.Convolution(net.relu4c, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.dsum3b = L.Eltwise(*[net.dinterp3a, net.deconv3b])
    net.drelu3b = L.ReLU(net.dsum3b, in_place=True)
    net.deconv3c = L.Convolution(net.drelu3b, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.drelu3c = L.ReLU(net.deconv3c, in_place=True)
    net.deconv4a = L.Deconvolution(net.drelu3c,
                                   convolution_param=dict(kernel_size=3,
                                                          num_output=32,
                                                          pad=1,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp4a = L.Interp(net.deconv4a, height=48, width=64)
    net.deconv4b = L.Convolution(net.relu3c, kernel_size=3, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    net.dsum4b = L.Eltwise(*[net.dinterp4a, net.deconv4b])
    net.drelu4b = L.ReLU(net.dsum4b, in_place=True)
    net.deconv4c = L.Convolution(net.drelu4b, kernel_size=3, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    net.drelu4c = L.ReLU(net.deconv4c, in_place=True)
    net.deconv5a = L.Deconvolution(net.deconv4c,
                                   convolution_param=dict(kernel_size=3,
                                                          num_output=16,
                                                          pad=1,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp5a = L.Interp(net.deconv5a, height=96, width=128)
    net.deconv5b = L.Convolution(net.relu2c, kernel_size=3, num_output=16, pad=1, weight_filler=dict(type='xavier'))
    net.dsum5b = L.Eltwise(*[net.dinterp5a, net.deconv5b])
    net.drelu5b = L.ReLU(net.dsum5b, in_place=True)
    net.deconv5c = L.Convolution(net.drelu5b, kernel_size=3, num_output=16, pad=1, weight_filler=dict(type='xavier'))

    net.conv8a = L.Convolution(net.deconv5c,
                               kernel_size=3,
                               num_output=32,
                               pad=1,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.conv8b = L.Convolution(net.conv8a, kernel_size=3, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    net.sum8b = L.Eltwise(*[net.conv8a, net.conv8b])
    net.relu8b = L.ReLU(net.sum8b, in_place=True)
    net.conv8c = L.Convolution(net.relu8b, kernel_size=3, num_output=32, pad=1, weight_filler=dict(type='xavier'))
    net.relu8c = L.ReLU(net.conv8c, in_place=True)
    net.conv9a = L.Convolution(net.relu8c,
                               kernel_size=3,
                               num_output=64,
                               pad=1,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.conv9b = L.Convolution(net.conv9a, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.sum9b = L.Eltwise(*[net.conv9a, net.conv9b])
    net.relu9b = L.ReLU(net.sum9b, in_place=True)
    net.conv9c = L.Convolution(net.relu9b, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.relu9c = L.ReLU(net.conv9c, in_place=True)
    net.conv10a = L.Convolution(net.relu9c,
                                kernel_size=5,
                                num_output=128,
                                pad=2,
                                stride=2,
                                weight_filler=dict(type='xavier'))
    net.conv10b = L.Convolution(net.conv10a, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.sum10b = L.Eltwise(*[net.conv10a, net.conv10b])
    net.relu10b = L.ReLU(net.sum10b, in_place=True)
    net.conv10c = L.Convolution(net.relu10b, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.relu10c = L.ReLU(net.conv10c, in_place=True)
    net.conv11a = L.Convolution(net.relu10c,
                                kernel_size=5,
                                num_output=256,
                                pad=2,
                                stride=2,
                                weight_filler=dict(type='xavier'))
    net.conv11b = L.Convolution(net.conv11a, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.sum11b = L.Eltwise(*[net.conv11a, net.conv11b])
    net.relu11b = L.ReLU(net.sum11b, in_place=True)
    net.conv11c = L.Convolution(net.relu11b, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.relu11c = L.ReLU(net.conv11c, in_place=True)
    net.conv12 = L.Convolution(net.relu11c,
                               kernel_size=5,
                               num_output=512,
                               pad=2,
                               stride=2,
                               weight_filler=dict(type='xavier'))
    net.fc13 = L.InnerProduct(net.conv12, num_output=2048, weight_filler=dict(type='gaussian', std=0.01))
    net.fc14a = L.InnerProduct(net.fc13, num_output=1024, weight_filler=dict(type='gaussian', std=0.01))
    net.relu14a = L.ReLU(net.fc14a, in_place=True)
    net.fc14b = L.InnerProduct(net.relu14a, num_output=768, weight_filler=dict(type='gaussian', std=0.01))
    net.reshape14 = L.Reshape(net.fc14b, reshape_param={'shape': {'dim': [n, 256, 3]}})

    net.deconv6a = L.Deconvolution(net.conv12,
                                   convolution_param=dict(kernel_size=5,
                                                          num_output=256,
                                                          pad=2,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp6a = L.Interp(net.deconv1a, height=6, width=8)
    net.deconv6b = L.Convolution(net.relu11c, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.dsum6b = L.Eltwise(*[net.dinterp6a, net.deconv6b])
    net.drelu6b = L.ReLU(net.dsum6b, in_place=True)
    net.deconv6c = L.Convolution(net.drelu6b, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    net.drelu6c = L.ReLU(net.deconv6c, in_place=True)
    net.deconv7a = L.Deconvolution(net.drelu6c,
                                   convolution_param=dict(kernel_size=5,
                                                          num_output=128,
                                                          pad=2,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp7a = L.Interp(net.deconv7a, height=12, width=16)
    net.deconv7b = L.Convolution(net.relu10c, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.dsum7b = L.Eltwise(*[net.dinterp7a, net.deconv7b])
    net.drelu7b = L.ReLU(net.dsum7b, in_place=True)
    net.deconv7c = L.Convolution(net.drelu7b, kernel_size=3, num_output=128, pad=1, weight_filler=dict(type='xavier'))
    net.drelu7c = L.ReLU(net.deconv7c, in_place=True)
    net.deconv8a = L.Deconvolution(net.drelu7c,
                                   convolution_param=dict(kernel_size=3,
                                                          num_output=64,
                                                          pad=1,
                                                          stride=2,
                                                          weight_filler=dict(type='xavier')))
    net.dinterp8a = L.Interp(net.deconv8a, height=24, width=32)
    net.deconv8b = L.Convolution(net.relu9c, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.dsum8b = L.Eltwise(*[net.dinterp8a, net.deconv8b])
    net.drelu8b = L.ReLU(net.dsum8b, in_place=True)
    net.deconv8c = L.Convolution(net.drelu8b, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.drelu8c = L.ReLU(net.deconv8c, in_place=True)
    net.deconv8d = L.Convolution(net.drelu8c, kernel_size=3, num_output=64, pad=1, weight_filler=dict(type='xavier'))
    net.drelu8d = L.ReLU(net.deconv8d, in_place=True)
    net.conv15 = L.Convolution(net.drelu8d, kernel_size=3, num_output=3, pad=1)
    net.reshape15 = L.Reshape(net.conv15, reshape_param={'shape': {'dim': [n, 768, 3]}})
    net.ptcloud = L.Concat(*[net.reshape14, net.reshape15])

    net_prototxt = net.to_proto()
    with open(NET_PROTOTXT_FILE_PATH, 'w') as f:
        f.write(str(net_prototxt))

    return NET_PROTOTXT_FILE_PATH
