import torch
import numpy as np
import abc


class ClassificationMetric(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self):
        return NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def compute(y_true: torch.Tensor, y_pred: torch.Tensor) -> dtype:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def accumulate(self, prev, new):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.compute(y_true, y_pred)


class Correct(ClassificationMetric):
    def name(self):
        return 'Correct'

    def dtype(self):
        return torch.int32

    def compute(self, y_true, y_pred):
        return (y_true == y_pred).count_nonzero()

    def accumulate(self, prev, new):
        return prev + new


class Accuracy(ClassificationMetric):
    def name(self):
        return 'Accuracy'

    def dtype(self):
        return np.float32

    def compute(self, y_true, y_pred):
        value = (y_true == y_pred).count_nonzero() / y_true.shape[0]
        value = value.detach().cpu().numpy().astype(self.dtype())
        return value