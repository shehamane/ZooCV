import torch

from trainer.trainer import Trainer


class SimpleTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device='cuda'):
        super(SimpleTrainer, self).__init__(model, loss_fn, optimizer, device)

    def train(self, loader):
        self.model.train()

        for batch, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * X.shape[0]
                print(f'Loss: {loss:>7f}')

    def test(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch, (ims, labels_true) in enumerate(loader):
                ims, labels_true = ims.to(self.device), labels_true.to(self.device)
                logits = self.model(ims)
                labels_pred = logits.argmax(1)
                total_loss += self.loss_fn(logits, labels_true)
                correct += (logits.argmax(1) == labels_true).type(torch.float).sum().item()
        avg_loss = total_loss / len(loader.dataset)
        print(f'Loss: {avg_loss:>7f}')

        for metric in self.metrics:
            metric_val = metric(labels_true, labels_pred)
            print(f'{metric.name}: {metric.format(metric_val)}')

