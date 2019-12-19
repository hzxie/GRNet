# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-15 20:33:52
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-19 13:26:38
# @Email:  cshzxie@gmail.com

import torch

import gridding


class GriddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale, ptcloud):
        grid, grid_pt_weights, grid_pt_indexes = gridding.forward(-scale, scale - 1, -scale, scale - 1, -scale,
                                                                  scale - 1, ptcloud)
        # print(grid.size())             # torch.Size(batch_size, n_grid_vertices)
        # print(grid_pt_weights.size())  # torch.Size(batch_size, n_pts, 8, 3)
        # print(grid_pt_indexes.size())  # torch.Size(batch_size, n_pts, 8)
        ctx.save_for_backward(grid_pt_weights, grid_pt_indexes)

        return grid

    @staticmethod
    def backward(ctx, grad_grid):
        grid_pt_weights, grid_pt_indexes = ctx.saved_tensors
        grad_ptcloud = gridding.backward(grid_pt_weights, grid_pt_indexes, grad_grid)
        # print(grad_ptcloud.size())   # torch.Size(batch_size, n_pts, 3)

        return None, grad_ptcloud


class Gridding(torch.nn.Module):
    def __init__(self, scale=1):
        super(Gridding, self).__init__()
        self.scale = scale // 2

    def forward(self, ptcloud):
        ptcloud = ptcloud * self.scale
        _ptcloud = torch.split(ptcloud, 1, dim=0)
        grids = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            grids.append(GriddingFunction.apply(self.scale, p))

        return torch.cat(grids, dim=0).contiguous()


class GriddingReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale, grid):
        ptcloud = gridding.rev_forward(scale, grid)
        ctx.save_for_backward(torch.Tensor([scale]), grid, ptcloud)
        return ptcloud

    @staticmethod
    def backward(ctx, grad_ptcloud):
        scale, grid, ptcloud = ctx.saved_tensors
        scale = int(scale.item())
        grad_grid = gridding.rev_backward(ptcloud, grid, grad_ptcloud)
        grad_grid = grad_grid.view(-1, scale, scale, scale)
        return None, grad_grid


class GriddingReverse(torch.nn.Module):
    def __init__(self, scale=1):
        super(GriddingReverse, self).__init__()
        self.scale = scale

    def forward(self, grid):
        ptcloud = GriddingReverseFunction.apply(self.scale, grid)
        return ptcloud / self.scale * 2


class GriddingDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, min_x, max_x, min_y, max_y, min_z, max_z, pred_cloud, gt_cloud):
        pred_grid, pred_grid_pt_weights, pred_grid_pt_indexes = gridding.forward(min_x, max_x, min_y, max_y, min_z,
                                                                                 max_z, pred_cloud)
        # print(pred_grid.size())             # torch.Size(batch_size, n_grid_vertices)
        # print(pred_grid_pt_weights.size())  # torch.Size(batch_size, n_pred_pts, 8, 3)
        # print(pred_grid_pt_indexes.size())  # torch.Size(batch_size, n_pred_pts, 8)

        gt_grid, gt_grid_pt_weights, gt_grid_pt_indexes = gridding.forward(min_x, max_x, min_y, max_y, min_z, max_z,
                                                                           gt_cloud)
        # print(gt_grid.size())               # torch.Size(batch_size, n_grid_vertices)
        # print(gt_grid_pt_weights.size())    # torch.Size(batch_size, n_gt_pts, 8, 3)
        # print(gt_grid_pt_indexes.size())    # torch.Size(batch_size, n_gt_pts, 8)

        ctx.save_for_backward(pred_grid_pt_weights, pred_grid_pt_indexes, gt_grid_pt_weights, gt_grid_pt_indexes)

        return pred_grid, gt_grid

    @staticmethod
    def backward(ctx, grad_pred_grid, grad_gt_grid):
        pred_grid_pt_weights, pred_grid_pt_indexes, gt_grid_pt_weights, gt_grid_pt_indexes = ctx.saved_tensors

        grad_pred_cloud = gridding.backward(pred_grid_pt_weights, pred_grid_pt_indexes, grad_pred_grid)
        # print(grad_ptcloud.size())   # torch.Size(batch_size, n_pred_pts, 3)
        grad_gt_cloud = gridding.backward(gt_grid_pt_weights, gt_grid_pt_indexes, grad_gt_grid)
        # print(grad_gt_cloud.size())  # torch.Size(batch_size, n_gt_pts, 3)

        return None, None, None, None, None, None, grad_pred_cloud, grad_gt_cloud


class GriddingDistance(torch.nn.Module):
    def __init__(self, scale=1):
        super(GriddingDistance, self).__init__()
        self.scale = scale

    def forward(self, pred_cloud, gt_cloud):
        '''
        pred_cloud(b, n_pts1, 3)
        gt_cloud(b, n_pts2, 3)
        '''
        pred_cloud = pred_cloud * self.scale
        gt_cloud = gt_cloud * self.scale

        min_pred_x = torch.min(pred_cloud[:, :, 0])
        max_pred_x = torch.max(pred_cloud[:, :, 0])
        min_pred_y = torch.min(pred_cloud[:, :, 1])
        max_pred_y = torch.max(pred_cloud[:, :, 1])
        min_pred_z = torch.min(pred_cloud[:, :, 2])
        max_pred_z = torch.max(pred_cloud[:, :, 2])

        min_gt_x = torch.min(gt_cloud[:, :, 0])
        max_gt_x = torch.max(gt_cloud[:, :, 0])
        min_gt_y = torch.min(gt_cloud[:, :, 1])
        max_gt_y = torch.max(gt_cloud[:, :, 1])
        min_gt_z = torch.min(gt_cloud[:, :, 2])
        max_gt_z = torch.max(gt_cloud[:, :, 2])

        min_x = torch.floor(torch.min(min_pred_x, min_gt_x)) - 1
        max_x = torch.ceil(torch.max(max_pred_x, max_gt_x)) + 1
        min_y = torch.floor(torch.min(min_pred_y, min_gt_y)) - 1
        max_y = torch.ceil(torch.max(max_pred_y, max_gt_y)) + 1
        min_z = torch.floor(torch.min(min_pred_z, min_gt_z)) - 1
        max_z = torch.ceil(torch.max(max_pred_z, max_gt_z)) + 1

        _pred_clouds = torch.split(pred_cloud, 1, dim=0)
        _gt_clouds = torch.split(gt_cloud, 1, dim=0)
        pred_grids = []
        gt_grids = []
        for pc, gc in zip(_pred_clouds, _gt_clouds):
            non_zeros = torch.sum(pc, dim=2).ne(0)
            pc = pc[non_zeros].unsqueeze(dim=0)
            non_zeros = torch.sum(gc, dim=2).ne(0)
            gc = gc[non_zeros].unsqueeze(dim=0)
            pred_grid, gt_grid = GriddingDistanceFunction.apply(min_x, max_x, min_y, max_y, min_z, max_z, pc, gc)
            pred_grids.append(pred_grid)
            gt_grids.append(gt_grid)

        return torch.cat(pred_grids, dim=0).contiguous(), torch.cat(gt_grids, dim=0).contiguous()


class GriddingLoss(torch.nn.Module):
    def __init__(self, scales=[], alphas=[]):
        super(GriddingLoss, self).__init__()
        self.scales = scales
        self.alphas = alphas
        self.gridding_dists = [GriddingDistance(scale=s) for s in scales]
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred_cloud, gt_cloud):
        gridding_loss = None
        n_dists = len(self.scales)

        for i in range(n_dists):
            scale = self.scales[i]
            alpha = self.alphas[i]
            gdist = self.gridding_dists[i]
            pred_grid, gt_grid = gdist(pred_cloud, gt_cloud)

            if gridding_loss is None:
                gridding_loss = alpha * self.l1_loss(pred_grid, gt_grid)
            else:
                gridding_loss += alpha * self.l1_loss(pred_grid, gt_grid)

        return gridding_loss
