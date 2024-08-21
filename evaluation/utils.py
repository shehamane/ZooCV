from abc import ABC


class Result(ABC):
    def __init__(self):
        return


class ClassificationResult(Result):
    def __init__(self, label, conf):
        super().__init__()
        self.label = label
        self.conf = conf
