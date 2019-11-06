# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-09 17:47:34
# @Email:  cshzxie@gmail.com

import torch
import torch.nn.functional as F

from extensions.permutohedral_layer import PermutohedralLayer


class RPLNet(torch.nn.Module):
    def __init__(self, cfg):
        super(RPLNet, self).__init__()
        self.cfg = cfg
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(3, 128, kernel_size=1),
            torch.nn.BatchNorm1d(128),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, kernel_size=1),
            torch.nn.BatchNorm1d(256),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(512, 512, kernel_size=1),
            torch.nn.BatchNorm1d(512),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(512, 1024, kernel_size=1),
            torch.nn.BatchNorm1d(1024),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.Linear(1024, 1024),
            torch.nn.Linear(1024, 3072),
        )

    def _pick_and_scale(self, points, scale_factor):
        return points * scale_factor

    def forward(self, data):
        partial_cloud = data['partial_cloud']
        # print(partial_cloud.size())     # torch.Size([batch_size, 3, 2048])
        features1 = self.conv1(partial_cloud)
        # print(features1.size())         # torch.Size([batch_size, 128, 2048])
        features2 = self.conv2(features1)
        # print(features2.size())         # torch.Size([batch_size, 256, 2048])
        features3 = torch.max(features2, dim=2, keepdim=True)[0].repeat(1, 1, 2048)
        # print(features3.size())         # torch.Size([batch_size, 256, 2048])
        features3 = torch.cat([features2, features3], dim=1)
        # print(features3.size())         # torch.Size([batch_size, 512, 2048])
        features4 = self.conv3(features3)
        # print(features4.size())         # torch.Size([batch_size, 512, 2048])
        features5 = self.conv4(features3)
        # print(features5.size())         # torch.Size([batch_size, 1024, 2048])
        features5 = torch.max(features5, dim=2)[0]
        # print(features5.size())         # torch.Size([batch_size, 1024])
        ptcloud = self.conv5(features5)
        # print(ptcloud.size())           # torch.Size([batch_size, 3072])

        return ptcloud.view(-1, 3, 1024)
