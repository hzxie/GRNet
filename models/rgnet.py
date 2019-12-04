# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-04 20:14:09
# @Email:  cshzxie@gmail.com

import torch
import torch.nn.functional as F

from extensions.gridding import Gridding, GriddingReverse


class RGNet(torch.nn.Module):
    def __init__(self, cfg):
        super(RGNet, self).__init__()
        self.cfg = cfg
        self.gridding = Gridding(scale=32)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc4 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU()
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU()
        )
        self.dconv6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=32)

    def forward(self, data):
        partial_cloud = data['partial_cloud']
        # print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        pt_features_32_l = self.gridding(partial_cloud).view(-1, 1, 32, 32, 32)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 1, 32, 32, 32])
        pt_features_16_l = self.conv1(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 32, 16, 16, 16])
        pt_features_8_l = self.conv2(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 64, 8, 8, 8])
        pt_features_4_l = self.conv3(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 128, 4, 4, 4])
        features = self.fc4(pt_features_4_l.view(-1, 8192))
        # print(features.size())          # torch.Size([batch_size, 2048])
        pt_features_4_r = self.fc5(features).view(-1, 128, 4, 4, 4) + pt_features_4_l
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 128, 4, 4, 4])
        pt_features_8_r = self.dconv6(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 64, 8, 8, 8])
        pt_features_16_r = self.dconv7(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 32, 16, 16, 16])
        pt_features_32_r = self.dconv8(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 1, 32, 32, 32])
        ptcloud = self.gridding_rev(pt_features_32_r.squeeze(dim=1))
        # print(ptcloud.size())           # torch.Size([batch_size, 32768, 3])

        return ptcloud
