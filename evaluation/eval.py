import abc
from typing import List, Type

import torch
from torch.utils.data import DataLoader

from evaluation.logging import Logger, Printer
from evaluation.metrics import Metric, ArtifactDrawer, ClassificationMetricCalculator
from evaluation.utils import ClassificationResult
from model.model import ClassificationModel, Model


class Evaluator(abc.ABC):
    def __init__(self,
                 logger: Logger = None,
                 device='cuda'):
        self.device = torch.device(device)
        if logger is None:
            self.logger = Printer()

    @abc.abstractmethod
    def evaluate(self,
                 model: Model,
                 loader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.modules.loss._Loss,
                 eval_id=None) -> None:
        raise NotImplementedError


class ClassificationEvaluator(Evaluator):
    def evaluate(self,
                 model: ClassificationModel,
                 loader: DataLoader,
                 loss_fn: torch.nn.modules.loss._Loss,
                 eval_id=1) -> None:
        model.nn.eval()
        metrics_calculator = ClassificationMetricCalculator()

        preds = {}
        gts = {}

        im_id = 0
        total_loss = 0
        with torch.no_grad():
            for it, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits_b = model.nn(images)
                total_loss += loss_fn(logits_b, labels)

                for logits, label_gt in zip(logits_b, labels):
                    preds[im_id] = ClassificationResult(logits.unsqueeze(0))
                    gts[im_id] = ClassificationResult(torch.nn.functional.one_hot(label_gt, num_classes=logits.shape[0]).unsqueeze(0))
                    im_id += 1

        gt, tp, accuracy = metrics_calculator.calculate(preds, gts)
        metrics = {'Accuracy': accuracy}
        avg_loss = (total_loss / len(loader)).item()
        self.logger.log_loss(eval_id, avg_loss)
        self.logger.log_metrics(eval_id, metrics)
