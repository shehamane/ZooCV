import abc
from typing import List

import torch
import pandas as pd

from evaluation.metrics import Metric


class Evaluator(abc.ABC):
    def __init__(self, freq, device='cuda', extensive=False, plot=True, save_to=None):
        self.freq = freq
        self.extensive = extensive
        self.plot = plot
        self.save_to = save_to
        self.device = torch.device(device)

    @abc.abstractmethod
    def evaluate(self, model, loader, loss_fn):
        raise NotImplementedError


class ClassificationEvaluator(Evaluator):
    def evaluate(self, model, loader, loss_fn):
        model.eval()

        gt = len(loader.dataset)
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for it, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = model(images)
                total_loss += loss_fn(logits, labels)

                labels_pred = logits.argmax(1)
                correct += labels_pred.eq(labels).int().sum().item()
        accuracy = correct / gt
        avg_loss = (total_loss / gt).detach().cpu().item()

        return {Metric.GT: gt, Metric.TP: correct, Metric.Accuracy: accuracy, Metric.loss: avg_loss}


class Logger:
    def __init__(self, metrics: List[Metric]):
        columns = ({Metric.loss: pd.Series(dtype=float), } |
                   {metric: pd.Series(dtype=float) for metric in metrics})
        index = pd.Index([0, ], dtype=int)
        self.log = pd.DataFrame(index=index, data=columns)

    def log_metrics(self, it, metrics):
        for name, value in metrics.items():
            if name in self.log.columns:
                self.log.loc[it, name] = value

    def print_metrics(self, it, metrics):
        print(f'Iteration {it}: {",".join([":".join((str(k), str(round(v, 3)))) for k, v in metrics.items()])}')
