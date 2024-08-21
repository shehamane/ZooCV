import abc
from typing import List

import torch
from torch.utils.data import DataLoader

from evaluation.logging import Logger, Printer
from evaluation.metrics import Metric, ArtifactDrawer, ClassificationMetricCalculator
from model.model import ClassificationModel


class Evaluator(abc.ABC):
    def __init__(self,
                 metrics: List[Metric],
                 artifacts: List[ArtifactDrawer] = None,
                 logger: Logger = None,
                 device='cuda'):
        self.metrics = metrics
        self.artifacts = artifacts
        self.device = torch.device(device)
        if logger is None:
            self.logger = Printer(metrics)
        self.runs = 0

    @abc.abstractmethod
    def evaluate(self, model: torch.nn.Module,
                 loader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.modules.loss._Loss,
                 eval_id=None):
        raise NotImplementedError


class ClassificationEvaluator(Evaluator):
    def evaluate(self,
                 model: ClassificationModel,
                 loader: DataLoader,
                 loss_fn: torch.nn.modules.loss._Loss,
                 eval_id=None):
        if eval_id is None:
            eval_id = self.runs

        model.nn.eval()
        metrics_calculator = ClassificationMetricCalculator(num_classes=model.nc)
        metrics_calculator.check_compatibility(self.metrics)

        preds = {}
        gt = {}
        im_idx = 0

        total_loss = 0
        results = {}
        with torch.no_grad():
            for it, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = model.nn(images)
                total_loss += loss_fn(logits, labels)

                labels_pred = logits.argmax(1).detach().cpu().numpy()
                labels_gt = labels.detach().cpu().numpy()

                for label_pred, label_gt in zip(labels_pred, labels_gt):
                    preds[im_idx] = label_pred
                    gt[im_idx] = label_gt
                    im_idx += 1

        metrics, metrics_per_class = metrics_calculator(gt, preds)
        self.logger.log_metrics(eval_id, metrics)
        self.runs += 1
