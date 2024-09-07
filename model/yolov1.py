import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from model.model import DetectionModel


class YoloV1(DetectionModel):
    def __init__(self, nc, grid_size=7, preds_per_cell=2):
        self.grid_size = grid_size
        self.preds_per_cell = preds_per_cell
        super().__init__('../config/darknet-v1.yaml', nc, S=grid_size, B=preds_per_cell)

    def get_loss(self):
        return YoloLoss(self.grid_size, self.preds_per_cell, self.nc)


class YoloLoss(_Loss):
    def __init__(self, grid_size, preds_per_cell, nc, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.grid_size = grid_size
        self.preds_per_cell = preds_per_cell
        self.nc = nc

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # predictions should be got directly from DarknetV1 last layer output
        # targets are in common YOLO format (batch, N, 5), every row contains class_id, x_rcenter, y_rcenter, w_r, h_r

        predictions = predictions.reshape((-1, self.grid_size, self.grid_size, self.nc + self.preds_per_cell * 5))

        for s_i in range(self.grid_size):
            for s_j in range(self.grid_size):
                s_x, s_y = s_i / self.grid_size, s_j / self.grid_size
                s_end_x, s_end_y = (s_i + 1) / self.grid_size, (s_j + 1) / self.grid_size
                for batch_i in range(targets.shape[0]):
                    grid_idx = torch.logical_and(
                        torch.logical_and(targets[batch_i, :, 1] > s_x,
                                          targets[batch_i, :, 2] > s_y),
                        torch.logical_and(targets[batch_i, :, 1] + targets[batch_i, :, 3] < s_end_x,
                                          targets[batch_i, :, 2] + targets[batch_i, :, 4] < s_end_y)
                    )
                    grid_targets = targets[batch_i, grid_idx]
                    grid_preds = predictions[batch_i, s_i, s_j]
                    grid_preds = torch.hstack((grid_preds[self.nc:].reshape(self.preds_per_cell, 5),
                                              grid_preds[:self.nc].expand(self.preds_per_cell, -1)))

                    preds_areas = grid_preds[:, 2] * grid_preds[:, 3]
                    targets_areas = grid_targets[:, 3] * grid_targets[:, 4]

                    inter_x1 = grid_targets[:, 1].expand(grid_preds.shape[0], -1)
                    inter_x1 = torch.maximum(grid_preds[:, 0], inter_x1.T)
                    inter_y1 = grid_targets[:, 2].expand(grid_preds.shape[1], -1)
                    inter_y1 = torch.maximum(grid_preds[:, 1], inter_y1.T)
                    inter_x2 = (grid_targets[:, 1] + grid_targets[:, :3]/2).expand(grid_preds.shape[0], -1)
                    inter_x2 = torch.maximum(grid_preds[:, 0] + grid_preds[:, 2]/2, inter_x2.T)
                    inter_y2 = (grid_targets[:, 20] + grid_targets[:, :4]/2).expand(grid_preds.shape[0], -1)
                    inter_y2 = torch.maximum(grid_preds[:, 1] + grid_preds[:, 3]/2, inter_y2.T)

                    inter_w = torch.maximum(0.0, inter_x2 - inter_x1)
                    inter_h = torch.maximum(0.0, inter_y2 - inter_y1)
                    inter_areas = inter_w * inter_h

                    preds_areas = preds_areas.expand(targets_areas.shape[0], -1)
                    targets_areas = targets_areas.expand(preds_areas.shape[0], -1)
                    outer_areas = preds_areas + targets_areas.T - inter_areas.T

                    ious = inter_areas.T / outer_areas
                    max_ious, argmax_ious = torch.max(ious, axis=1)

