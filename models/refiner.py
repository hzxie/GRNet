# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-17 18:49:45
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-20 15:55:25
# @Email:  cshzxie@gmail.com

import torch

import utils.helpers


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU()
        )
        self.fc3 = torch.nn.Linear(64, 24)

    def forward(self, sparse_cloud, point_features):
        # print(sparse_cloud.size())      # torch.Size([batch_size, 2048, 3])
        # print(point_features.size())    # torch.Size([batch_size, 2048, 256])
        point_features = self.fc1(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 256])
        point_features = self.fc2(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 64])
        point_offset = self.fc3(point_features).view(-1, 16384, 3)
        # print(point_features.size())    # torch.Size([batch_size, 16384, 3])
        sparse_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, 16384, 3)
        # print(sparse_cloud.size())      # torch.Size([batch_size, 16384, 3])

        return sparse_cloud + point_offset