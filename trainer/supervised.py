import numpy as np
import torch

from trainer.trainer import Trainer


class ClassificationTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device='cuda', metrics=None):
        super(ClassificationTrainer, self).__init__(model, loss_fn, optimizer, device)

    def train(self, loader):
        self.model.train()

        for it, (images, labels) in enumerate(loader):
            images, labels = images.to(self.device), labels.to(self.device)

            logits_pred = self.model(images)
            loss = self.loss_fn(logits_pred, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.logging and it % self.log_freq == 0:
                loss_log = (loss / loader.batch_size).detach().cpu().numpy().astype(np.float32)
                metrics = self.compute_metrics(logits_pred, labels)
                self.logger.log_loss(it, loss_log)
                self.logger.log_metrics(it, metrics)

    def test(self, loader):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for batch, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                labels_pred = logits.argmax(1)
                total_loss += self.loss_fn(logits, labels)

    def compute_metrics(self, logits_pred, labels_gt):
        metrics = {}
        for metric in self.metrics:
            metrics[metric.name] = metric.compute(labels_gt, logits_pred)
        return metrics
