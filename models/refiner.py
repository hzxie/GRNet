# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-17 18:49:45
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 16:42:11
# @Email:  cshzxie@gmail.com

import torch

import utils.helpers


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.conv1 = torch.nn.Conv1d(2053, 1024, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(1024, 512, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(512, 3, kernel_size=1)

    def forward(self, sparse_cloud, global_features):
        batch_size = sparse_cloud.size(0)

        grid = torch.meshgrid(torch.linspace(-0.05, 0.05, 4), torch.linspace(-0.05, 0.05, 4))
        grid = torch.stack(grid, dim=2).view(-1, 2).unsqueeze(dim=0).repeat(batch_size, 2048, 1)
        grid = utils.helpers.var_or_cuda(grid)
        # print(grid.size())            # torch.Size([batch_size, 32768, 2])
        point_features = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 16, 1).view(-1, 32768, 3)
        # print(point_features.size())  # torch.Size([batch_size, 32768, 3])
        global_features = global_features.unsqueeze(dim=1).repeat(1, 32768, 1)
        # print(global_features.size()) # torch.Size([batch_size, 32768, 2048])
        features = torch.cat([grid, point_features, global_features], dim=2).permute(0, 2, 1).contiguous()
        # print(features.size())        # torch.Size([batch_size, 2053, 32768])
        features = self.conv1(features)
        # print(features.size())        # torch.Size([batch_size, 1024, 32768])
        features = self.conv2(features)
        # print(features.size())        # torch.Size([batch_size, 512, 32768])
        features = self.conv3(features)
        # print(features.size())        # torch.Size([batch_size, 3, 32768])
        dense_ptcloud = features + point_features.permute(0, 2, 1)
        # print(features.size())        # torch.Size([batch_size, 3, 32768])

        return dense_ptcloud.permute(0, 2, 1).contiguous()