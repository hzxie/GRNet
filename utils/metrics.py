# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-09-06 11:41:57
# @Email:  cshzxie@gmail.com

import open3d
import torch

from extensions.chamfer_dist import ChamferDistance


class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'ChamferDistance',
        'eval_func': 'cls._get_chamfer_distance',
        'eval_object': ChamferDistance(),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt):
        _values = [0] * len(cls.ITEMS)
        for i, item in enumerate(cls.ITEMS):
            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        pred = cls._get_open3d_ptcloud(pred)
        gt = cls._get_open3d_ptcloud(gt)

        dist1 = open3d.compute_point_cloud_to_point_cloud_distance(pred, gt)
        dist2 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))
        return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.PointCloud()
        ptcloud.points = open3d.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distance(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        dist1, dist2 = chamfer_distance(pred, gt)
        return (torch.mean(dist1) + torch.mean(dist2)).item() * 1000

    @classmethod
    def items(cls):
        return [i['name'] for i in cls.ITEMS]

    def __init__(self, metric_name, values):
        self._metric_name = metric_name
        self._values = [item['init_value'] for item in Metrics.ITEMS]

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            item_indexes = {}
            for idx, item in enumerate(Metrics.ITEMS):
                item_name = item['name']
                item_indexes[item_name] = idx
            for k, v in values.items():
                self._values[item_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(Metrics.ITEMS)):
            item = self.ITEMS[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(Metrics.ITEMS):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if self._is_greater_better else _value < other_value
