import torch
from torch.utils.data import DataLoader

from evaluation.eval import Evaluator
from evaluation.logging import Logger
from model.model import ClassificationModel
from trainer.trainer import Trainer


class SimpleTrainer(Trainer):
    def __init__(
            self,
            model: ClassificationModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_fn: torch.nn.modules.loss._Loss,
            optimizer: torch.optim.Optimizer,
            evaluator: Evaluator = None,
            logger: Logger = None,
            eval_freq: int = 1000,
            log_freq: int = 10,
            device: str = 'cuda',
    ):
        super().__init__(model, train_loader, val_loader, loss_fn, optimizer, evaluator, logger, eval_freq, log_freq, device)

    def train(self) -> None:
        self.model.nn.train()

        for batch_it, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)

            preds = self.model.nn(images)
            loss = self.loss_fn(preds, labels)

            loss.backward()
            self.optimizer.step()

            if self.evaluator is not None and self.it % self.eval_freq == 0:
                    self.evaluator.evaluate(self.model, self.val_loader, self.loss_fn, self.it)
                    self.model.nn.train()

            if self.logger is not None and self.it % self.log_freq == 0:
                batch_loss = loss.item()
                if self.logger is not None:
                    self.logger.log_loss(self.it, batch_loss)
            self.it += 1
