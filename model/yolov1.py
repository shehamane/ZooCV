import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from model.model import DetectionModel


class YoloV1(DetectionModel):
    def __init__(self, config, num_classes, grid_size=7, preds_per_cell=2, conf_thresh=0.5):
        super().__init__(config, num_classes, conf_thresh)
        self.grid_size = self.config['vars']['S'] = grid_size
        self.preds_per_cell = self.config['vars']['B'] = preds_per_cell


class YoloV1Loss(_Loss):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        pass
