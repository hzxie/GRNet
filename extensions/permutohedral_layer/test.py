# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np
import sys
import torch
import unittest

sys.path.append('../')
import permutohedral_layer

BATCH_SIZE = 1
N_INPUT_CHANNELS = 2
HEIGHT = 5
WIDTH = 6


def get_filter_size(neighborhood_size, in_channels):
    return (neighborhood_size + 1)**(in_channels + 1) - neighborhood_size**(in_channels + 1)


class PermutohedralLayerTest(unittest.TestCase):
    def setUp(self):
        self.data = torch.from_numpy(
            np.array([
                0, 0, 1.56894, 1.71588, 0, 0, 1.56518, 1.53057, 0.76686, -0.592583, 0.534071, -1.02683, -0.604775,
                1.68742, -0.89592, -1.50184, 0.506595, 0.135528, -0.48416, 0.412273, 0.503575, 0.507338, 0.456594,
                1.02362, 0, 0, 0.929329, 0.4986, 0, 0, 0, 0, -0.642905, 0.437683, 0, 0, 0.542442, -1.87837, 0.816111,
                0.461546, 0.353122, 1.43353, -0.519231, 0.114483, -0.439219, -0.238896, 0.309394, -1.6441, -0.875012,
                0.186592, -0.92626, 0.243145, -0.0358702, 1.23521, 0, 0, 0.585628, -0.843697, 0, 0
            ]).reshape(BATCH_SIZE, N_INPUT_CHANNELS, HEIGHT, WIDTH)).float().cuda()
        self.features = torch.from_numpy(
            np.array([
                0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0, 0.2, 0.4, 0.6, 0.8, 1, 0, 0.2, 0.4, 0.6, 0.8, 1, 0, 0.2, 0.4,
                0.6, 0.8, 1, 0, 0.2, 0.4, 0.6, 0.8, 1, 0, 0.2, 0.4, 0.6, 0.8, 1
            ]).reshape(BATCH_SIZE, N_INPUT_CHANNELS, HEIGHT, WIDTH)).float().cuda()

    def test_forward(self):
        NEIGHBORHOOD_SIZE = 3
        GROUP = 1
        DO_SKIP_BLUR = False
        USE_BIAS_TERM = False
        N_OUTPUT_CHANNELS = 1

        weights = torch.nn.Parameter(
            torch.from_numpy(
                np.array([
                    0.1323, 0.6974, 0.0385, 0.1482, 0.9143, 0.3580, 0.7855, 0.6853, 0.7724, 0.8354, 0.5875, 0.0648,
                    0.2228, 0.0221, 0.1667, 0.3385, 0.5716, 0.8733, 0.7973, 0.6162, 0.8735, 0.4067, 0.4518, 0.5935,
                    0.1076, 0.4054, 0.7348, 0.6255, 0.6997, 0.1674, 0.6014, 0.1302, 0.2214, 0.1487, 0.9410, 0.7894,
                    0.3460, 0.1162, 0.4337, 0.6263, 0.8687, 0.3424, 0.3681, 0.9897, 0.1241, 0.3494, 0.5519, 0.0626,
                    0.4717, 0.1418, 0.0482, 0.4744, 0.4467, 0.5200, 0.8097, 0.0078, 0.0759, 0.0823, 0.4734, 0.3025,
                    0.4921, 0.2631, 0.3241, 0.3063, 0.0510, 0.0217, 0.4484, 0.5637, 0.1158, 0.9030, 0.3043, 0.4858,
                    0.0309, 0.9056
                ])).reshape(N_OUTPUT_CHANNELS, N_INPUT_CHANNELS // GROUP, 1,
                            get_filter_size(NEIGHBORHOOD_SIZE, N_INPUT_CHANNELS)).float()).cuda()
        bias = torch.nn.Parameter(torch.zeros(1, 1, 1, N_OUTPUT_CHANNELS)).cuda()
        bias_multiplier = torch.nn.Parameter(torch.ones(1, 1, 1, HEIGHT * WIDTH)).cuda()

        expected_output = [
            0.141078, 0.160562, 0.179919, 0.199151, 0.218258, 0.237243, 0.157174, 0.149898, 0.16911, 0.1882, 0.207166,
            0.232033, 0.17601, 0.146237, 0.1585, 0.177448, 0.197525, 0.226493, 0.198352, 0.162737, 0.148082, 0.166892,
            0.191128, 0.220592, 0.225278, 0.182046, 0.151396, 0.158187, 0.184329, 0.214291
        ]
        actual_output = permutohedral_layer.PermutohedralFunction.apply(NEIGHBORHOOD_SIZE, GROUP, N_OUTPUT_CHANNELS,
                                                                        DO_SKIP_BLUR, USE_BIAS_TERM, weights, bias,
                                                                        bias_multiplier, self.data, self.features).view(-1)
        for i in range(len(expected_output)):
            assert abs(expected_output[i] - actual_output[i].item()) < 1e-6

if __name__ == '__main__':
    unittest.main()
