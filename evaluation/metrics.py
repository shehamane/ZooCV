from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Tuple

import numpy as np


class Metric(Enum):
    GT = 1
    TP = 2
    FP = 3
    FN = 4
    Accuracy = 5
    Precision = 6
    Recall = 7
    F1 = 8
    AP_50 = 9
    AP_50_95 = 10
    mAP = 12


def metrics2names(metrics: List[Metric]) -> List[str]:
    return list(map(lambda m: m.name, metrics))


class MetricCalculator(ABC):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @abstractmethod
    def calculated_metrics(self) -> List[Metric]:
        raise NotImplementedError

    def __call__(self, gt: Dict[str, np.ndarray], preds: Dict[str, np.ndarray]):
        raise NotImplementedError


class ClassificationMetricCalculator(MetricCalculator):
    def calculated_metrics(self) -> List[Metric]:
        return [Metric.GT, Metric.FP, Metric.TP, Metric.FN,
                Metric.Accuracy, Metric.Recall, Metric.Precision,
                Metric.F1]

    def check_compatibility(self, metrics: List[Metric]):
        for metric_name in metrics2names(metrics):
            if metric_name not in metrics2names(self.calculated_metrics()):
                raise Exception(f'{self.__class__.__name__} can\'t calculate metric: {metric_name}')

    def calculate(self, gt: Dict[Any, int], preds: Dict[Any, int]) \
            -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        metrics = {metric: 0.0 for metric in metrics2names(self.calculated_metrics())}
        metrics_per_class = {label: {metric: 0.0 for metric in metrics2names(self.calculated_metrics())}
                             for label in range(self.num_classes)}
        if len(gt) == 0:
            return metrics, metrics_per_class

        metrics[Metric.GT.name] = len(gt)
        for im_id in gt.keys():
            label_gt = gt[im_id]
            label_pred = preds[im_id]
            metrics_per_class[label_gt][Metric.GT.name] += 1
            if label_pred == label_gt:
                metrics[Metric.TP.name] += 1
                metrics_per_class[label_pred][Metric.TP.name] += 1
            else:
                metrics[Metric.FP.name] += 1
                metrics[Metric.FN.name] += 1
                metrics_per_class[label_pred][Metric.FP.name] += 1
                metrics_per_class[label_gt][Metric.FN.name] += 1

        metrics[Metric.Accuracy.name] = metrics[Metric.TP.name] / metrics[Metric.GT.name]
        metrics[Metric.Precision.name] = metrics[Metric.TP.name] / (metrics[Metric.TP.name] + metrics[Metric.FP.name])
        metrics[Metric.Recall.name] = metrics[Metric.TP.name] / metrics[Metric.GT.name]
        metrics[Metric.F1.name] = 2 * metrics[Metric.Precision.name] * metrics[Metric.Recall.name] / (
                + metrics[Metric.Precision.name] + metrics[Metric.Recall.name])
        for label in range(self.num_classes):
            metrics_per_class[label][Metric.Accuracy.name] = (metrics_per_class[label][Metric.TP.name] /
                                                              metrics_per_class[label][Metric.GT.name])
            if metrics_per_class[label][Metric.TP.name] + metrics_per_class[label][Metric.FP.name] == 0:
                metrics_per_class[label][Metric.Precision.name] = 0
            else:
                metrics_per_class[label][Metric.Precision.name] = (metrics_per_class[label][Metric.TP.name] /
                                                                   (metrics_per_class[label][Metric.TP.name] +
                                                                    metrics_per_class[label][Metric.FP.name]))
            metrics_per_class[label][Metric.Recall.name] = (metrics_per_class[label][Metric.TP.name] /
                                                            metrics_per_class[label][Metric.GT.name])
            if metrics_per_class[label][Metric.Recall.name] == 0 and metrics_per_class[label][
                Metric.Precision.name] == 0:
                metrics_per_class[label][Metric.F1.name] = 0
            else:
                metrics_per_class[label][Metric.F1.name] = (2 * metrics_per_class[label][Metric.Precision.name] *
                                                            metrics_per_class[label][Metric.Recall.name]) / (
                                                                   + metrics_per_class[label][Metric.Precision.name] +
                                                                   metrics_per_class[label][Metric.Recall.name])

        return metrics, metrics_per_class

    def __call__(self, gt: Dict[Any, int], preds: Dict[Any, int]) \
            -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        return self.calculate(gt, preds)


class DetectionMetricCalculator(MetricCalculator):
    def calculated_metrics(self) -> List[Metric]:
        return [Metric.GT, Metric.FP, Metric.TP, Metric.FN,
                Metric.Accuracy, Metric.Recall, Metric.Precision,
                Metric.F1]


class ArtifactDrawer(ABC):
    @abstractmethod
    def __call__(self, metrics):
        raise NotImplementedError
