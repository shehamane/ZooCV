import abc
from typing import List

import numpy as np
import pandas as pd
import torch

from metric.classification import ClassificationMetric


class Trainer(abc.ABC):
    def __init__(
            self, model: torch.nn.Module,
            loss_fn,
            optimizer: torch.optim.optimizer.Optimizer,
            device: str,
            metrics: List[ClassificationMetric] = None,
            log_freq=100,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.metrics = metrics

        if log_freq:
            self.logging = True
            self.log_freq = log_freq
            self.logger = Logger(metrics)
        else:
            self.logging = False

    @abc.abstractmethod
    def train(self, loader: torch.utils.data.DataLoader):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self, loader: torch.utils.data.DataLoader):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_metrics(self, pred, gt):
        raise NotImplementedError


class Logger:
    def __init__(self, metrics: List[ClassificationMetric]):
        columns = ({'loss': pd.Series(dtype=float), } |
                   {metric.name: pd.Series(dtype=metric.dtype) for metric in metrics})
        index = pd.Index([0, ], dtype=np.int32)
        self.log = pd.DataFrame(index=index, data=columns)

    def log_loss(self, it, loss):
        self.log.loc[it, 'loss'] = loss

    def log_metrics(self, it, metrics):
        for name, value in metrics:
            self.log.loc[it, name] = value
