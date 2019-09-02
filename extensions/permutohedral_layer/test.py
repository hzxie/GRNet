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

    def test_forward_without_bias_1(self):
        NEIGHBORHOOD_SIZE = 3
        N_OUTPUT_CHANNELS = 1
        GROUP = 1
        DO_SKIP_BLUR = False
        USE_BIAS_TERM = False

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
                                                                        bias_multiplier, self.data,
                                                                        self.features).view(-1)
        for i in range(len(expected_output)):
            assert abs(expected_output[i] - actual_output[i].item()) < 1e-6

    def test_forward_without_bias_2(self):
        NEIGHBORHOOD_SIZE = 3
        N_OUTPUT_CHANNELS = 2
        GROUP = 1
        DO_SKIP_BLUR = False
        USE_BIAS_TERM = False

        weights = torch.nn.Parameter(
            torch.from_numpy(
                np.array([
                    0.5021, 0.0808, 0.1300, 0.0037, 0.0900, 0.7380, 0.5003, 0.5992, 0.8590, 0.5942, 0.2779, 0.4272,
                    0.2538, 0.0785, 0.8325, 0.8073, 0.2169, 0.6298, 0.0537, 0.7440, 0.0398, 0.2114, 0.0280, 0.6852,
                    0.8143, 0.3317, 0.9143, 0.9252, 0.0743, 0.1257, 0.0342, 0.4272, 0.1521, 0.8283, 0.4482, 0.0655,
                    0.0039, 0.0522, 0.1913, 0.3830, 0.4114, 0.8568, 0.7087, 0.1507, 0.6350, 0.1647, 0.7555, 0.1491,
                    0.6394, 0.0879, 0.0163, 0.0635, 0.5957, 0.8960, 0.9698, 0.7643, 0.5854, 0.3660, 0.0755, 0.7076,
                    0.0769, 0.8268, 0.9251, 0.2357, 0.4510, 0.8530, 0.3781, 0.4296, 0.4140, 0.2051, 0.3979, 0.5243,
                    0.0386, 0.4116, 0.3210, 0.1167, 0.4434, 0.6002, 0.9113, 0.4777, 0.6002, 0.1779, 0.7167, 0.5999,
                    0.5278, 0.8685, 0.7545, 0.6099, 0.6418, 0.4805, 0.7603, 0.3483, 0.9601, 0.8955, 0.0394, 0.5211,
                    0.8051, 0.1423, 0.9552, 0.3077, 0.3644, 0.7126, 0.8151, 0.4037, 0.9388, 0.6032, 0.3575, 0.4161,
                    0.3829, 0.6212, 0.3134, 0.0478, 0.6631, 0.6889, 0.0525, 0.6416, 0.1704, 0.4762, 0.2783, 0.2961,
                    0.6712, 0.5909, 0.9193, 0.4749, 0.7097, 0.3195, 0.0790, 0.5283, 0.1373, 0.8415, 0.6499, 0.7637,
                    0.8716, 0.0146, 0.8270, 0.0501, 0.7787, 0.5214, 0.3531, 0.1292, 0.0786, 0.4242, 0.0910, 0.3798,
                    0.4713, 0.0436, 0.1656, 0.1864
                ])).reshape(N_OUTPUT_CHANNELS, N_INPUT_CHANNELS // GROUP, 1,
                            get_filter_size(NEIGHBORHOOD_SIZE, N_INPUT_CHANNELS)).float()).cuda()
        bias = torch.nn.Parameter(torch.zeros(1, 1, 1, N_OUTPUT_CHANNELS)).cuda()
        bias_multiplier = torch.nn.Parameter(torch.ones(1, 1, 1, HEIGHT * WIDTH)).cuda()

        expected_output = [
            0.103097, 0.100565, 0.0980489, 0.0955495, 0.0930663, 0.090599, 0.109696, 0.130067, 0.12748, 0.124909,
            0.122354, 0.12228, 0.117418, 0.153899, 0.156371, 0.153732, 0.152072, 0.155967, 0.126578, 0.164483,
            0.184737, 0.182032, 0.185473, 0.191856, 0.137617, 0.176868, 0.204695, 0.212958, 0.220979, 0.230172,
            0.192484, 0.165642, 0.138974, 0.112479, 0.0861561, 0.0600023, 0.207423, 0.187871, 0.161378, 0.135056,
            0.108903, 0.0894282, 0.224906, 0.207818, 0.183372, 0.15722, 0.133422, 0.120717, 0.245642, 0.223959,
            0.204966, 0.178983, 0.164944, 0.154052, 0.270632, 0.242848, 0.22315, 0.206141, 0.198452, 0.18964
        ]
        actual_output = permutohedral_layer.PermutohedralFunction.apply(NEIGHBORHOOD_SIZE, GROUP, N_OUTPUT_CHANNELS,
                                                                        DO_SKIP_BLUR, USE_BIAS_TERM, weights, bias,
                                                                        bias_multiplier, self.data,
                                                                        self.features).view(-1)
        for i in range(len(expected_output)):
            assert abs(expected_output[i] - actual_output[i].item()) < 1e-6

    def test_forward_without_bias_3(self):
        NEIGHBORHOOD_SIZE = 3
        N_OUTPUT_CHANNELS = 2
        GROUP = 2
        DO_SKIP_BLUR = False
        USE_BIAS_TERM = False

        weights = torch.nn.Parameter(
            torch.from_numpy(
                np.array([
                    0.9563, 0.9710, 0.1391, 0.7237, 0.0513, 0.9796, 0.5293, 0.8683, 0.9887, 0.3814, 0.5046, 0.0728,
                    0.6375, 0.8009, 0.7446, 0.4081, 0.4133, 0.7653, 0.7872, 0.3806, 0.9470, 0.4092, 0.7763, 0.7680,
                    0.5838, 0.5031, 0.2360, 0.6857, 0.0998, 0.0508, 0.3512, 0.4108, 0.2213, 0.9582, 0.9080, 0.8887,
                    0.1852, 0.5220, 0.4328, 0.5508, 0.1733, 0.1334, 0.0827, 0.5330, 0.1629, 0.1145, 0.4290, 0.6580,
                    0.9343, 0.4470, 0.0170, 0.6574, 0.7234, 0.7337, 0.4713, 0.2950, 0.3095, 0.9189, 0.6696, 0.4937,
                    0.1802, 0.8292, 0.5326, 0.3017, 0.9982, 0.8356, 0.4158, 0.2481, 0.3167, 0.0646, 0.3026, 0.0480,
                    0.5483, 0.4559
                ])).reshape(N_OUTPUT_CHANNELS, N_INPUT_CHANNELS // GROUP, 1,
                            get_filter_size(NEIGHBORHOOD_SIZE, N_INPUT_CHANNELS)).float()).cuda()
        bias = torch.nn.Parameter(torch.zeros(1, 1, 1, N_OUTPUT_CHANNELS)).cuda()
        bias_multiplier = torch.nn.Parameter(torch.ones(1, 1, 1, HEIGHT * WIDTH)).cuda()

        expected_output = [
            0.339508, 0.350793, 0.362005, 0.373145, 0.384212, 0.395208, 0.327138, 0.343477, 0.354609, 0.365669,
            0.376659, 0.395246, 0.312663, 0.335108, 0.347348, 0.35833, 0.371036, 0.395287, 0.295495, 0.322394,
            0.340219, 0.351124, 0.370336, 0.39533, 0.274803, 0.307516, 0.330708, 0.347095, 0.369591, 0.395376,
            -0.0264046, -0.0252216, -0.0240462, -0.0228785, -0.0217183, -0.0205656, -0.0275501, -0.0262135, -0.0250458,
            -0.0238856, -0.0227329, -0.0260976, -0.0288906, -0.0272228, -0.026027, -0.0248743, -0.0249181, -0.0319799,
            -0.0304805, -0.0284323, -0.0269904, -0.0258451, -0.0305557, -0.0382468, -0.0323967, -0.0298478, -0.0280408,
            -0.0292291, -0.0365484, -0.0449373
        ]
        actual_output = permutohedral_layer.PermutohedralFunction.apply(NEIGHBORHOOD_SIZE, GROUP, N_OUTPUT_CHANNELS,
                                                                        DO_SKIP_BLUR, USE_BIAS_TERM, weights, bias,
                                                                        bias_multiplier, self.data,
                                                                        self.features).view(-1)
        for i in range(len(expected_output)):
            assert abs(expected_output[i] - actual_output[i].item()) < 1e-6

    def test_forward_with_bias(self):
        NEIGHBORHOOD_SIZE = 3
        N_OUTPUT_CHANNELS = 2
        GROUP = 2
        DO_SKIP_BLUR = False
        USE_BIAS_TERM = True

        weights = torch.nn.Parameter(
            torch.from_numpy(
                np.array([
                    0.9563, 0.9710, 0.1391, 0.7237, 0.0513, 0.9796, 0.5293, 0.8683, 0.9887, 0.3814, 0.5046, 0.0728,
                    0.6375, 0.8009, 0.7446, 0.4081, 0.4133, 0.7653, 0.7872, 0.3806, 0.9470, 0.4092, 0.7763, 0.7680,
                    0.5838, 0.5031, 0.2360, 0.6857, 0.0998, 0.0508, 0.3512, 0.4108, 0.2213, 0.9582, 0.9080, 0.8887,
                    0.1852, 0.5220, 0.4328, 0.5508, 0.1733, 0.1334, 0.0827, 0.5330, 0.1629, 0.1145, 0.4290, 0.6580,
                    0.9343, 0.4470, 0.0170, 0.6574, 0.7234, 0.7337, 0.4713, 0.2950, 0.3095, 0.9189, 0.6696, 0.4937,
                    0.1802, 0.8292, 0.5326, 0.3017, 0.9982, 0.8356, 0.4158, 0.2481, 0.3167, 0.0646, 0.3026, 0.0480,
                    0.5483, 0.4559
                ])).reshape(N_OUTPUT_CHANNELS, N_INPUT_CHANNELS // GROUP, 1,
                            get_filter_size(NEIGHBORHOOD_SIZE, N_INPUT_CHANNELS)).float()).cuda()
        bias = torch.nn.Parameter(
            torch.from_numpy(np.array([0.0818, 0.8894])).reshape(1, 1, 1, N_OUTPUT_CHANNELS).float()).cuda()
        bias_multiplier = torch.nn.Parameter(
            torch.from_numpy(
                np.array([
                    0.4784, 0.9353, 0.7067, 0.7665, 0.6245, 0.8526, 0.0751, 0.7218, 0.8618, 0.1907, 0.0649, 0.9583,
                    0.6774, 0.1310, 0.1830, 0.2861, 0.7582, 0.1888, 0.3619, 0.5874, 0.5918, 0.9045, 0.4876, 0.7668,
                    0.9014, 0.7652, 0.8306, 0.0115, 0.3857, 0.2601
                ])).reshape(1, 1, 1, HEIGHT * WIDTH).float()).cuda()

        expected_output = [
            0.378641, 0.427301, 0.419813, 0.435844, 0.435296, 0.464951, 0.333282, 0.40252, 0.425104, 0.381268,
            0.381968, 0.473635, 0.368075, 0.345824, 0.362317, 0.381733, 0.433057, 0.41073, 0.325098, 0.370444,
            0.388628, 0.425112, 0.410221, 0.458054, 0.348538, 0.37011, 0.398652, 0.348035, 0.401142, 0.416652,
            0.399084, 0.806634, 0.604493, 0.658847, 0.533712, 0.737737, 0.0392438, 0.615755, 0.741439, 0.145723,
            0.0349892, 0.826214, 0.573589, 0.0892886, 0.136733, 0.229583, 0.649425, 0.135939, 0.291393, 0.494001,
            0.499357, 0.778617, 0.403116, 0.643745, 0.769309, 0.650721, 0.710695, -0.019001, 0.306493, 0.186396
        ]
        actual_output = permutohedral_layer.PermutohedralFunction.apply(NEIGHBORHOOD_SIZE, GROUP, N_OUTPUT_CHANNELS,
                                                                        DO_SKIP_BLUR, USE_BIAS_TERM, weights, bias,
                                                                        bias_multiplier, self.data,
                                                                        self.features).view(-1)
        for i in range(len(expected_output)):
            assert abs(expected_output[i] - actual_output[i].item()) < 1e-6


if __name__ == '__main__':
    unittest.main()
