# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import numpy as np
import random
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters']
            self.transformers.append({'callback': transformer(parameters), 'objects': tr['objects']})

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = random.random()    # Random values for random crop and random flip
            for k, v in data.items():
                if k in objects and k in data:
                    if transform.__class__ in [RandomCrop, RandomFlip]:
                        data[k] = transform(v, rnd_value)
                    else:
                        data[k] = transform(v)

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        tensor = None
        shape = arr.shape
        if len(shape) == 2:    # Point Clouds
            tensor = arr
        elif len(shape) == 3:    # RGB/Depth Images
            tensor = arr.transpose(2, 0, 1)
        else:
            raise Exception('Unknown shape: %s' % list(shape))
        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(tensor.copy()).float()


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']

    def __call__(self, img):
        img = img.astype(np.float32) / 255.
        img /= self.std
        img -= self.mean
        img *= 2

        return img


class CenterCrop(object):
    def __init__(self, parameters):
        self.img_size_h = parameters['img_size'][0]
        self.img_size_w = parameters['img_size'][1]
        self.crop_size_h = parameters['crop_size'][0]
        self.crop_size_w = parameters['crop_size'][1]

    def __call__(self, img):
        img_w, img_h, _ = img.shape
        x_left = (img_w - self.crop_size_w) * .5
        x_right = x_left + self.crop_size_w
        y_top = (img_h - self.crop_size_h) * .5
        y_bottom = y_top + self.crop_size_h

        # Crop the image
        img = cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        img = img[..., np.newaxis] if len(img.shape) == 2 else img

        return img


class RandomCrop(object):
    def __init__(self, parameters):
        self.img_size_h = parameters['img_size'][0]
        self.img_size_w = parameters['img_size'][1]
        self.crop_size_h = parameters['crop_size'][0]
        self.crop_size_w = parameters['crop_size'][1]

    def __call__(self, img, rnd_value):
        img_w, img_h, _ = img.shape
        x_left = (img_w - self.crop_size_w) * rnd_value
        x_right = x_left + self.crop_size_w
        y_top = (img_h - self.crop_size_h) * rnd_value
        y_bottom = y_top + self.crop_size_h

        # Crop the image
        img = cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        img = img[..., np.newaxis] if len(img.shape) == 2 else img

        return img


class RandomFlip(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img, rnd_value):
        if rnd_value > 0.5:
            img = np.fliplr(img)

        return img


class RandomPermuteRGB(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img):
        rgb_permutation = np.random.permutation(3)
        return img[..., rgb_permutation]


class RandomBackground(object):
    def __init__(self, parameters):
        self.random_bg_color_range = parameters['bg_color']

    def __call__(self, img):
        img_h, img_w, img_c = img.shape
        if not img_c == 4:
            return img

        r, g, b = [
            np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1) for i in range(3)
        ]
        alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
        img = img[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        img = alpha * bg_color + (1 - alpha) * img

        return img


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.choice(len(ptcloud), self.n_points, replace=False)
        return ptcloud[choice, :]
