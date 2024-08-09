import abc

import torch

from evaluation.evaluator import Evaluator, Logger


class Trainer(abc.ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader,
            val_loader,
            loss_fn,
            optimizer: torch.optim.Optimizer,
            evaluator: Evaluator,
            logger: Logger,
            print_log: bool = True,
            device: str = 'cuda',
            plotter=None
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.evaluator = evaluator
        self.logger = logger
        self.last_iter = 0
        self.print_log = print_log

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

