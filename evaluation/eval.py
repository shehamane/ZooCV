import abc
from typing import List, Type

import torch
from torch.utils.data import DataLoader

from evaluation.logging import Logger, Printer
from evaluation.metrics import Metric, ArtifactDrawer, ClassificationMetricCalculator
from model.model import ClassificationModel, Model


class Evaluator(abc.ABC):
    def __init__(self,
                 metrics: List[Metric],
                 artifacts: List[ArtifactDrawer] = None,
                 logger: Type[Logger] = Printer,
                 device='cuda'):
        self.metrics = metrics
        self.artifacts = artifacts
        self.device = torch.device(device)
        self.logger = logger(self.metrics)

    @abc.abstractmethod
    def evaluate(self, model: Model,
                 loader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.modules.loss._Loss,
                 eval_id=None):
        raise NotImplementedError


class ClassificationEvaluator(Evaluator):
    def evaluate(self,
                 model: ClassificationModel,
                 loader: DataLoader,
                 loss_fn: torch.nn.modules.loss._Loss,
                 eval_id=1):
        model.nn.eval()
        metrics_calculator = ClassificationMetricCalculator(num_classes=model.nc)
        metrics_calculator.check_compatibility(self.metrics)

        preds = {}
        gts = {}

        im_id = 0
        total_loss = 0
        with torch.no_grad():
            for it, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = model.nn(images)
                total_loss += loss_fn(logits, labels)

                labels_pred = logits.argmax(1).detach().cpu().numpy()
                labels_gt = labels.detach().cpu().numpy()

                for label_pred, label_gt in zip(labels_pred, labels_gt):
                    preds[im_id] = label_pred
                    gts[im_id] = label_gt
                    im_id += 1

        metrics, metrics_per_class = metrics_calculator(gts, preds)
        avg_loss = (total_loss / len(loader.dataset)).item()
        self.logger.log_loss(eval_id, avg_loss)
        self.logger.log_metrics(eval_id, metrics)
