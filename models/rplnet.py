# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-06 16:58:33
# @Email:  cshzxie@gmail.com

import torch
import torch.nn.functional as F

from extensions.permutohedral_layer import PermutohedralLayer


class RPLNet(torch.nn.Module):
    def __init__(self, cfg):
        super(RPLNet, self).__init__()
        self.cfg = cfg
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(3, 32, kernel_size=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )
        self.conv2 = PermutohedralLayer(
            in_channels_data=32,
            in_channels_feature=3,
            out_channels=64,
            out_height=1,
            out_width=2048)
        self.conv2p = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv3 = PermutohedralLayer(
            in_channels_data=64,
            in_channels_feature=3,
            out_channels=128,
            out_height=1,
            out_width=2048)
        self.conv3p = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.conv4 = PermutohedralLayer(
            in_channels_data=128,
            in_channels_feature=3,
            out_channels=256,
            out_height=1,
            out_width=2048)
        self.conv4p = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.conv5 = PermutohedralLayer(
            in_channels_data=256,
            in_channels_feature=3,
            out_channels=256,
            out_height=1,
            out_width=2048)
        self.conv5p = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.conv6 = PermutohedralLayer(
            in_channels_data=256,
            in_channels_feature=3,
            out_channels=256,
            out_height=1,
            out_width=2048)
        self.conv6p = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv1d(960, 240, kernel_size=1),
            torch.nn.BatchNorm1d(240),
            torch.nn.ReLU()
        )
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv1d(240, 60, kernel_size=1),
            torch.nn.BatchNorm1d(60),
            torch.nn.ReLU()
        )
        self.conv9 = torch.nn.Sequential(
            torch.nn.Conv1d(60, 3, kernel_size=1),
            torch.nn.BatchNorm1d(3),
            torch.nn.ReLU()
        )

    def _pick_and_scale(self, points, scale_factor):
        return points * scale_factor

    def forward(self, data):
        partial_cloud = data['partial_cloud'].permute(0, 2, 1)
        # print(partial_cloud.size())
        features1 = self.conv1(partial_cloud)
        # print(features1.size())
        features2 = self.conv2p(self.conv2(
            features1.unsqueeze(dim=2),
            self._pick_and_scale(partial_cloud, 64).unsqueeze(dim=2)
        ))
        # print(features2.size())
        features3 = self.conv3p(self.conv3(
            features2.unsqueeze(dim=2),
            self._pick_and_scale(partial_cloud, 32).unsqueeze(dim=2)
        ))
        # print(features3.size())
        features4 = self.conv4p(self.conv4(
            features3.unsqueeze(dim=2),
            self._pick_and_scale(partial_cloud, 16).unsqueeze(dim=2)
        ))
        # print(features4.size())
        features5 = self.conv5p(self.conv5(
            features4.unsqueeze(dim=2),
            self._pick_and_scale(partial_cloud, 8).unsqueeze(dim=2)
        ))
        # print(features5.size())
        features6 = self.conv6p(self.conv6(
            features5.unsqueeze(dim=2),
            self._pick_and_scale(partial_cloud, 4).unsqueeze(dim=2)
        ))
        # print(features6.size())
        features = torch.cat([features2, features3, features4, features5, features6], dim=1).squeeze(dim=2)
        # print(features.size())
        ptcloud = self.conv9(self.conv8(self.conv7(features)))
        # print(ptcloud.size())

        return ptcloud
