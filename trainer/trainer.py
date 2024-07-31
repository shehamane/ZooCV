import abc

import torch


class Trainer(abc.ABC):
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.metrics = []

    @abc.abstractmethod
    def train(self, loader):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self, loader):
        raise NotImplementedError

    def set_metrics(self, metrics):
        self.metrics = metrics
