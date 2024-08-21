import abc

import torch
from torch.utils.data import DataLoader

from evaluation.eval import Evaluator, Logger
from evaluation.logging import Printer
from model.model import Model


class Trainer(abc.ABC):
    def __init__(
            self,
            model: Model,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_fn: torch.nn.modules.loss._Loss,
            optimizer: torch.optim.Optimizer,
            evaluator: Evaluator,
            logger: Logger = None,
            eval_freq: int = 1000,
            device: str = 'cuda',
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model.nn = self.model.nn.to(self.device)
        self.evaluator = evaluator
        self.last_iter = 0
        self.eval_freq = eval_freq
        if logger is None:
            self.logger = Printer([])

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
