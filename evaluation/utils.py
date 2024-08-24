from abc import ABC
from typing import Tuple, List

import torch


class Result(ABC):
    pass


class ClassificationResult(Result):
    def __init__(self, class_id, conf):
        self.class_id = class_id
        self.conf = conf


class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class DetectionResult(Result):
    # Takes 2D Tensor with shape (N, 5), one prediction looks like (cls, x1, y1, x2, y2, conf)

    def __init__(self, preds: torch.Tensor):
        self.bboxes: List[Tuple[int, BBox, float]] = []
        for pred in preds:
            pred_np = pred.detach().cpu().numpy()
            class_id = pred_np[0]
            bbox = BBox(pred_np[1], pred_np[2], pred_np[3], pred_np[4])
            conf = pred_np[5]
            self.bboxes.append((class_id, bbox, conf))
