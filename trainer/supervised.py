import torch
from torch.utils.data import DataLoader

from evaluation.eval import Evaluator
from evaluation.logging import Logger
from model.model import ClassificationModel
from trainer.trainer import Trainer


class ClassificationTrainer(Trainer):
    def __init__(
            self,
            model: ClassificationModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_fn: torch.nn.modules.loss._Loss,
            optimizer: torch.optim.Optimizer,
            evaluator: Evaluator,
            logger: Logger = None,
            eval_freq: int = 1000,
            device: str = 'cuda',
    ):
        super().__init__(model, train_loader, val_loader, loss_fn, optimizer, evaluator, logger, eval_freq, device)

    def train(self):
        self.model.nn.train()

        total_loss = 0
        for batch_it, (images, labels) in enumerate(self.train_loader):
            it = self.last_iter + batch_it

            images, labels = images.to(self.device), labels.to(self.device)

            logits_pred = self.model.nn(images)
            loss = self.loss_fn(logits_pred, labels)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if it % self.eval_freq == 0:
                self.evaluator.evaluate(self.model, self.val_loader, self.loss_fn, it)
                self.model.nn.train()

        avg_loss = total_loss / len(self.train_loader.dataset)
        self.logger.log_loss(it, avg_loss)
        self.last_iter = it
