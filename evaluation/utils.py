from abc import ABC


class Result(ABC):
    def __init__(self, idx):
        self.idx = idx


class ClassificationResult(Result):
    def __init__(self, idx, class_id, conf):
        super().__init__(idx)
        self.class_id = class_id
        self.conf = conf
