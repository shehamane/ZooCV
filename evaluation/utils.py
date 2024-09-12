from abc import ABC, abstractmethod
from typing import Iterator
from torchtyping import TensorType

import torch


class Predict(ABC):
    pass


class ClassificationPredict(Predict):
    def __init__(self, probs: TensorType['C']):
        self.probs = probs

    @property
    def cls(self) -> int:
        return self.probs.argmax().item()


class BBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class DetectionPredict(Predict):
    def __init__(self, pred: TensorType[6]):
        self.pred = pred

    @property
    def bbox(self) -> BBox:
        return BBox(*self.pred[1:-1])

    @property
    def cls(self) -> int:
        return self.pred[0].item()

    @property
    def conf(self) -> float:
        return self.pred[-1].item()


class Result(ABC):
    def __init__(self, preds: TensorType):
        self.preds = preds

    def __len__(self):
        return self.preds.shape[0]

    @abstractmethod
    def __getitem__(self, key: int):
        raise NotImplementedError


class ClassificationResult(Result):
    def __init__(self, preds: TensorType['N', 'C']):
        super().__init__(preds)

    def __getitem__(self, key: int) -> ClassificationPredict:
        return ClassificationPredict(self.preds[key])

    def __iter__(self) -> Iterator[ClassificationPredict]:
        for i in range(self.preds.shape[0]):
            yield self[i]


class DetectionResult(Result):
    def __init__(self, preds: TensorType['N', 6]):
        super().__init__(preds)

    def __getitem__(self, key: int) -> DetectionPredict:
        return DetectionPredict(self.preds[key])

    def __iter__(self) -> Iterator[DetectionPredict]:
        for i in range(self.preds.shape[0]):
            yield self[i]
