# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-17 18:49:45
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-24 14:43:19
# @Email:  cshzxie@gmail.com

import torch

import utils.helpers


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1792, 1792),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1792, 448),
            torch.nn.ReLU()
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(448, 112),
            torch.nn.ReLU()
        )
        self.fc4 = torch.nn.Linear(112, 24)

    def forward(self, sparse_cloud, point_features):
        # print(sparse_cloud.size())      # torch.Size([batch_size, 2048, 3])
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc1(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc2(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 448])
        point_features = self.fc3(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 112])
        point_offset = self.fc4(point_features).view(-1, 16384, 3)
        # print(point_features.size())    # torch.Size([batch_size, 16384, 3])
        sparse_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, 16384, 3)
        # print(sparse_cloud.size())      # torch.Size([batch_size, 16384, 3])

        return sparse_cloud + point_offset