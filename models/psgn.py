# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-06 11:39:50
# @Email:  cshzxie@gmail.com
#
# References:
# - https://github.com/fanhqme/PointSetGeneration/blob/master/depthestimate/train_nn.py

import torch
import torch.nn.functional as F


class PSGN(torch.nn.Module):
    def __init__(self, cfg):
        super(PSGN, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.conv1a = torch.nn.Sequential(torch.nn.Conv2d(4, 16, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv1b = torch.nn.Sequential(torch.nn.Conv2d(16, 16, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv2a = torch.nn.Sequential(torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), torch.nn.ReLU())
        self.conv2b = torch.nn.Sequential(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv2c = torch.nn.Sequential(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv3a = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), torch.nn.ReLU())
        self.conv3b = torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv3c = torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv4a = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
                                          torch.nn.ReLU())
        self.conv4b = torch.nn.Sequential(torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv4c = torch.nn.Sequential(torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv5a = torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
                                          torch.nn.ReLU())
        self.conv5b = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv5c = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv6a = torch.nn.Sequential(torch.nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
                                          torch.nn.ReLU())
        self.conv6b = torch.nn.Sequential(torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv6c = torch.nn.Sequential(torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv6d = torch.nn.Sequential(torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv6e = torch.nn.Sequential(torch.nn.Conv2d(512, 512, kernel_size=5, padding=2, stride=2),
                                          torch.nn.ReLU())
        self.fc7 = torch.nn.Sequential(torch.nn.Linear(6144, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 2048))

        self.dconv1a = torch.nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv1b = torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.dconv1c = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.ReLU())
        self.dconv2a = torch.nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv2b = torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.dconv2c = torch.nn.Sequential(torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU())
        self.dconv3a = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dconv3b = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dconv3c = torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU())
        self.dconv4a = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dconv4b = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dconv4c = torch.nn.Sequential(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1), torch.nn.ReLU())
        self.dconv5a = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dconv5b = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.dconv5c = torch.nn.Sequential(torch.nn.Conv2d(16, 16, kernel_size=3, padding=1), torch.nn.ReLU())

        self.conv8a = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv8b = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv8c = torch.nn.Sequential(torch.nn.Conv2d(32, 32, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv9a = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv9b = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv9c = torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv10a = torch.nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.conv10b = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv10c = torch.nn.Sequential(torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv11a = torch.nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2)
        self.conv11b = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv11c = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.ReLU())
        self.conv12 = torch.nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2)
        self.fc13 = torch.nn.Linear(6144, 2048)

        self.dconv6a = torch.nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv6b = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.dconv6c = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.ReLU())
        self.dconv7a = torch.nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv7b = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dconv7c = torch.nn.Sequential(torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU())
        self.dconv8a = torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv8b = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dconv8c = torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU())
        self.dconv8d = torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU())

        self.fc14 = torch.nn.Sequential(torch.nn.Linear(2048, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 768))
        self.conv14 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, data):
        rgb_imgs = data['rgb_img']
        depth_imgs = data['depth_img']
        rgbd_imgs = torch.cat((rgb_imgs, depth_imgs), dim=1)

        features = self.conv1b(self.conv1a(rgbd_imgs))
        image_features0 = features
        # print(features.size())            # torch.Size([batch_size, 16, 192, 256])
        features = self.conv2c(self.conv2b(self.conv2a(features)))
        image_features1 = features
        # print(features.size())            # torch.Size([batch_size, 32, 96, 128])
        features = self.conv3c(self.conv3b(self.conv3a(features)))
        image_features2 = features
        # print(features.size())            # torch.Size([batch_size, 64, 48, 64])
        features = self.conv4c(self.conv4b(self.conv4a(features)))
        image_features3 = features
        # print(features.size())            # torch.Size([batch_size, 128, 24, 32])
        features = self.conv5c(self.conv5b(self.conv5a(features)))
        image_features4 = features
        # print(features.size())            # torch.Size([batch_size, 256, 12, 16])
        features = self.conv6c(self.conv6c(self.conv6b(self.conv6a(features))))
        image_features5 = features
        # print(features.size())            # torch.Size([batch_size, 512, 6, 8])
        features = self.conv6e(features)
        image_features6 = features
        # print(features.size())            # torch.Size([batch_size, 512, 3, 4])
        features = self.fc7(features.view(-1, 6144))
        image_features7 = features
        # print(features.size())            # torch.Size([batch_size, 2048])

        image_features6 = self.dconv1a(image_features6)
        # print(image_features6.size())     # torch.Size([batch_size, 256, 6, 8])
        image_features5 = self.dconv1b(image_features5)
        # print(image_features5.size())     # torch.Size([batch_size, 256, 6, 8])
        image_features5 = self.dconv1c(F.relu(image_features6 + image_features5))
        # print(image_features5.size())     # torch.Size([batch_size, 256, 6, 8])
        features = self.dconv2a(image_features5)
        # print(features.size())            # torch.Size([batch_size, 128, 12, 16])
        image_features4 = self.dconv2b(image_features4)
        # print(image_features4.size())     # torch.Size([batch_size, 128, 12, 16])
        image_features4 = self.dconv2c(F.relu(features + image_features4))
        # print(image_features4.size())     # torch.Size([batch_size, 128, 12, 16])
        features = self.dconv3a(image_features4)
        # print(features.size())            # torch.Size([batch_size, 64, 24, 32])
        image_features3 = self.dconv3b(image_features3)
        # print(image_features3.size())     # torch.Size([batch_size, 64, 24, 32])
        image_features3 = self.dconv3c(F.relu(features + image_features3))
        # print(image_features3.size())     # torch.Size([batch_size, 64, 24, 32])
        features = self.dconv4a(image_features3)
        # print(features.size())              # torch.Size([batch_size, 32, 48, 64])
        image_features2 = self.dconv4b(image_features2)
        # print(image_features2.size())     # torch.Size([batch_size, 32, 48, 64])
        image_features2 = self.dconv4c(F.relu(features + image_features2))
        # print(image_features2.size())     # torch.Size([batch_size, 32, 48, 64])
        features = self.dconv5a(image_features2)
        # print(features.size())            # torch.Size([batch_size, 16, 96, 128])
        image_features1 = self.dconv5b(image_features1)
        # print(image_features1.size())     # torch.Size([batch_size, 16, 96, 128])
        image_features1 = self.dconv5c(F.relu(features + image_features1))
        # print(image_features1.size())     # torch.Size([batch_size, 16, 96, 128])

        features = self.conv8a(image_features1)
        # print(features.size())            # torch.Size([batch_size, 32, 48, 64])
        image_features2 = self.conv8b(image_features2)
        # print(image_features2.size())     # torch.Size([batch_size, 32, 48, 64])
        image_features2 = self.conv8c(F.relu(features + image_features2))
        # print(image_features2.size())     # torch.Size([batch_size, 32, 48, 64])
        features = self.conv9a(image_features2)
        # print(features.size())            # torch.Size([batch_size, 64, 24, 32])
        image_features3 = self.conv9b(image_features3)
        # print(image_features3.size())     # torch.Size([batch_size, 64, 24, 32])
        image_features3 = self.conv9c(F.relu(features + image_features3))
        # print(image_features3.size())     # torch.Size([batch_size, 64, 24, 32])
        features = self.conv10a(image_features3)
        # print(features.size())            # torch.Size([batch_size, 128, 12, 16])
        image_features4 = self.conv10b(image_features4)
        # print(image_features4.size())     # torch.Size([batch_size, 128, 12, 16])
        image_features4 = self.conv10c(F.relu(features + image_features4))
        # print(image_features4.size())     # torch.Size([batch_size, 128, 12, 16])
        features = self.conv11a(image_features4)
        # print(features.size())            # torch.Size([batch_size, 256, 6, 8])
        image_features5 = self.conv11b(image_features5)
        # print(image_features5.size())     # torch.Size([batch_size, 256, 6, 8])
        image_features5 = self.conv11c(F.relu(features + image_features5))
        # print(image_features5.size())     # torch.Size([batch_size, 256, 6, 8])
        image_features6 = self.conv12(image_features5)
        # print(image_features6.size())     # torch.Size([batch_size, 512, 3, 4])
        features = self.fc13(image_features6.view(-1, 6144))
        image_features7 = F.relu(features + image_features7)
        # print(image_features7.size())     # torch.Size([batch_size, 2048])

        image_features6 = self.dconv6a(image_features6)
        # print(image_features6.size())     # torch.Size([batch_size, 256, 6, 8])
        image_features5 = self.dconv6b(image_features5)
        # print(image_features5.size())     # torch.Size([batch_size, 256, 6, 8])
        image_features5 = self.dconv6c(F.relu(image_features6 + image_features5))
        # print(image_features5.size())     # torch.Size([batch_size, 256, 6, 8])
        image_features5 = self.dconv7a(image_features5)
        # print(image_features5.size())     # torch.Size([batch_size, 128, 12, 16])
        image_features4 = self.dconv7b(image_features4)
        # print(image_features4.size())     # torch.Size([batch_size, 128, 12, 16])
        image_features4 = self.dconv7c(F.relu(image_features5 + image_features4))
        # print(image_features4.size())     # torch.Size([batch_size, 128, 12, 16])
        image_features4 = self.dconv8a(image_features4)
        # print(image_features4.size())     # torch.Size([batch_size, 64, 24, 32])
        image_features3 = self.dconv8b(image_features3)
        # print(image_features3.size())     # torch.Size([batch_size, 64, 24, 32])
        image_features3 = self.dconv8c(F.relu(image_features4 + image_features3))
        # print(image_features3.size())     # torch.Size([batch_size, 64, 24, 32])
        image_features3 = self.dconv8d(image_features3)
        # print(image_features3.size())     # torch.Size([batch_size, 64, 24, 32])

        image_features7 = self.fc14(image_features7).view(-1, 256, 3)
        # print(image_features7.size())     # torch.Size([batch_size, 256, 3])
        image_features3 = self.conv14(image_features3).view(-1, 768, 3)
        # print(image_features3.size())     # torch.Size([batch_size, 768, 3])
        generated_ptclouds = torch.cat((image_features7, image_features3), dim=1)
        # print(generated_ptclouds.size())  # torch.Size([batch_size, 1024, 3])

        return generated_ptclouds
