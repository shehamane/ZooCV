from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np

from evaluation.utils import Result, ClassificationResult


class Metric(ABC):
    def __init__(self):
        self.val = None

    @abstractmethod
    def calculate(self, preds: Dict[str, Result], gts: Dict[str, Result]) -> Union[int, float]:
        raise NotImplementedError

    def __call__(self, preds: Dict[str, Result], gts: Dict[str, Result]):
        if self.val is None:
            self.val = self.calculate(preds, gts)

        return self.val

    def item(self) -> Union[int, float]:
        return self.val


class GT(Metric):
    def calculate(self, preds: Dict[str, Result], gts: Dict[str, Result]) -> int:
        gt = 0
        for im_gts in gts.values():
            gt += len(im_gts)
        return gt


class ClassificationMetric(Metric):
    @abstractmethod
    def calculate(self, preds: Dict[str, ClassificationResult], gts: Dict[str, ClassificationResult]) \
            -> Union[int, float, List, Dict]:
        raise NotImplementedError


class ClassificationTP(ClassificationMetric):
    def calculate(self, preds: Dict[str, ClassificationResult], gts: Dict[str, ClassificationResult]) -> int:
        tp = 0

        for key in gts.keys():
            im_preds = preds[key]
            im_gts = gts[key]
            gt_label = im_gts[0].cls

            for pred in im_preds:
                if pred.cls == gt_label:
                    tp += 1

        self.val = tp
        return self.val


class Accuracy(ClassificationMetric):
    def __init__(self, true_positives: ClassificationTP, ground_truths: GT):
        super().__init__()
        self.tp = true_positives
        self.gt = ground_truths

    def calculate(self, preds: Dict[str, ClassificationResult], gts: Dict[str, ClassificationResult]) -> float:
        return self.tp(preds, gts) / self.gt(preds, gts)


class MetricCalculator(ABC):
    @abstractmethod
    def calculate(self, gt: Dict[str, Result], preds: Dict[str, Result]):
        raise NotImplementedError


class ClassificationMetricCalculator(MetricCalculator):
    def calculate(self, preds: Dict[str, ClassificationResult], gts: Dict[str, ClassificationResult]) \
            -> Tuple[int, int, float]:
        gt = GT()
        gt(preds, gts)

        tp = ClassificationTP()
        tp(preds, gts)

        accuracy = Accuracy(tp, gt)
        accuracy(preds, gts)

        return gt.item(), tp.item(), accuracy.item()


class DetectionMetricCalculator(MetricCalculator):
    pass


class ArtifactDrawer(ABC):
    pass
