from trainer.trainer import Trainer


class ClassificationTrainer(Trainer):
    def train(self):
        self.model.train()

        it = None
        for batch_it, (images, labels) in enumerate(self.train_loader):
            it = self.last_iter + batch_it

            images, labels = images.to(self.device), labels.to(self.device)

            logits_pred = self.model(images)
            loss = self.loss_fn(logits_pred, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if it % self.evaluator.freq == 0:
                    eval = self.evaluator.evaluate(self.model, self.val_loader, self.loss_fn)
                    if self.print_log:
                        self.logger.print_metrics(it, eval)
                    self.logger.log_metrics(it, eval)
                    self.model.train()
        self.last_iter = it
