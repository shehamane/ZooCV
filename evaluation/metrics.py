from enum import Enum


class Metric(Enum):
    loss = 0
    GT = 1
    TP = 2
    FP = 3
    FN = 4
    Accuracy = 5
    Precision = 6
    Recall = 7
    F1 = 8
    AP50 = 9
    AP75 = 10
    AP90 = 11
    MAP = 12
