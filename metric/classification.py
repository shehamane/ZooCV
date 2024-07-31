import torch
import abc


class ClassificationMetric(abc.ABC):
    @abc.abstractmethod
    def compute(self, y_true, y_pred):
        raise NotImplementedError

    @staticmethod
    def format(val):
        return val

    def __call__(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


class Accuracy(ClassificationMetric):
    name = 'Accuracy'

    def compute(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return (y_true == y_pred).count_nonzero() / y_true.shape[0]

    @staticmethod
    def format(val: float):
        return f'{val * 100:>3f}%'

