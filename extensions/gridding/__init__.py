# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-15 20:33:52
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-11-19 20:58:34
# @Email:  cshzxie@gmail.com

import torch

import gridding_dist


class GriddingDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_cloud, gt_cloud):
        min_pred_x = torch.min(pred_cloud[:, :, 0], dim=2)[0]
        max_pred_x = torch.max(pred_cloud[:, :, 0], dim=2)[0]
        min_pred_y = torch.min(pred_cloud[:, :, 1], dim=2)[0]
        max_pred_y = torch.max(pred_cloud[:, :, 1], dim=2)[0]
        min_pred_z = torch.min(pred_cloud[:, :, 2], dim=2)[0]
        max_pred_z = torch.max(pred_cloud[:, :, 2], dim=2)[0]

        min_gt_x = torch.min(gt_cloud[:, :, 0], dim=2)[0]
        max_gt_x = torch.max(gt_cloud[:, :, 0], dim=2)[0]
        min_gt_y = torch.min(gt_cloud[:, :, 1], dim=2)[0]
        max_gt_y = torch.max(gt_cloud[:, :, 1], dim=2)[0]
        min_gt_z = torch.min(gt_cloud[:, :, 2], dim=2)[0]
        max_gt_z = torch.max(gt_cloud[:, :, 2], dim=2)[0]

        min_x = torch.floor(torch.min(min_pred_x, min_gt_x))
        max_x = torch.ceil(torch.max(max_pred_x, max_gt_x))
        min_y = torch.floor(torch.min(min_pred_y, min_gt_y))
        max_y = torch.ceil(torch.max(max_pred_y, max_gt_y))
        min_z = torch.floor(torch.min(min_pred_z, min_gt_z))
        max_z = torch.ceil(torch.max(max_pred_z, max_gt_z))

        pred_grid_weights = gridding.forward(min_x, max_x, min_y, max_y, min_z, max_z, pred_cloud)
        # print(pred_grid_weights.size())   # torch.Size(batch_size, n_grid_vertices, n_pred_pts, 3)
        pred_grid = torch.prod(pred_grid_weights, dim=3)
        pred_grid = torch.sum(pred_grid, dim=2)

        gt_grid_weights = gridding.forward(min_x, max_x, min_y, max_y, min_z, max_z, gt_cloud)
        # print(pred_grid_weights.size())   # torch.Size(batch_size, n_grid_vertices, n_gt_pts, 3)
        gt_grid = torch.prod(gt_grid_weights, dim=3)
        gt_grid = torch.sum(gt_grid, dim=2)

        min_max_values = torch.Tensor([min_x, max_x, min_y, max_y, min_z, max_z])
        ctx.save_for_backward(min_max_values, pred_grid_weights, gt_grid_weights)
        return pred_grid, gt_grid

    @staticmethod
    def backward(ctx, grad_pred_grid, grad_gt_grid):
        min_max_values, pred_grid_weights, gt_grid_weights = ctx.saved_tensors
        min_x, max_x, min_y, max_y, min_z, max_z = min_max_values

        return grad_pt_cloud, grad_gt_cloud


class GriddingDistance(torch.nn.Module):
    def __init__(self, scale=1):
        super(GriddingDistance, self).__init__()
        self.scale = scale

    def forward(self, pred_cloud, gt_cloud):
        '''
        pred_cloud(b, n_pts1, 3)
        gt_cloud(b, n_pts2, 3)
        '''
        pred_cloud = pred_cloud / self.scale
        gt_cloud = gt_cloud / self.scale

        return GriddingLossFunction(pred_cloud, gt_cloud)
