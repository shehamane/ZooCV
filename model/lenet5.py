import torch

from evaluation.utils import ClassificationResult
from model.model import ClassificationModel


class LeNet5(ClassificationModel):
    def __call__(self, x: torch.Tensor) -> ClassificationResult:
        logits = self.nn(x).detach().cpu().numpy()
        label = logits.argmax(axis=1)
        conf = logits[label]
        return ClassificationResult(label, conf)
