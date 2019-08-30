import sys
sys.path.append('../')

import torch
import permutohedral_layer

NEIGHBORHOOD_SIZE = 3
GROUP = 1
BATCH_SIZE = 1
N_INPUT_CHANNELS = 2
N_OUTPUT_CHANNELS = 2
WIDTH = 6
HEIGHT = 5
DO_SKIP_BLUR = False

def get_filter_size(neighborhood_size, in_channels):
    return (neighborhood_size + 1)**(in_channels + 1) - neighborhood_size**(in_channels + 1)

data = torch.rand(BATCH_SIZE, N_INPUT_CHANNELS, HEIGHT, WIDTH).cuda()
features = torch.rand(BATCH_SIZE, N_OUTPUT_CHANNELS, HEIGHT, WIDTH).cuda()
weights = torch.nn.Parameter(
    torch.rand(N_OUTPUT_CHANNELS, N_INPUT_CHANNELS // GROUP, 1,
                 get_filter_size(NEIGHBORHOOD_SIZE, N_INPUT_CHANNELS))).cuda()
bias = torch.nn.Parameter(torch.zeros(1, 1, 1, N_OUTPUT_CHANNELS)).cuda()
bias_multiplier = torch.nn.Parameter(torch.ones(1, 1, 1, HEIGHT * WIDTH)).cuda()


# PermutohedralFunction.apply(
#   self.neighborhood_size, self.group, self.skip_blur, self.bias is not None,
#     data, features, features, self.bias, self.bias_multiplier)

permutohedral_layer.PermutohedralFunction.apply(
	NEIGHBORHOOD_SIZE, GROUP, DO_SKIP_BLUR, False, weights, bias, bias_multiplier, data, features)
