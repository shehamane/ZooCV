from abc import ABC

import yaml

from evaluation.utils import Result, ClassificationResult, DetectionResult
from utils.build import ModelBuilder


class Model(ABC):
    def __init__(self, config):
        with open(config) as config_f:
            self.config = yaml.safe_load(config_f)
            if 'vars' not in self.config:
                self.config['vars'] = {}
        self.nn = None

    def __call__(self, x) -> Result:
        raise NotImplementedError

    def parameters(self):
        return self.nn.parameters()


class ClassificationModel(Model, ABC):
    def __init__(self, config, nc):
        super().__init__(config)

        self.nc = self.config['vars']['nc'] = nc
        builder = ModelBuilder(self.config)
        self.nn = builder.build()

    def __call__(self, x) -> ClassificationResult:
        raise NotImplementedError


class DetectionModel(Model, ABC):
    def __init__(self, config, nc, **kwargs):
        super().__init__(config)

        self.nc = self.config['vars']['nc'] = nc
        for k, v in kwargs.items():
            self.config['vars'][k] = v
        builder = ModelBuilder(self.config)
        self.nn = builder.build()

    def __call__(self, x) -> DetectionResult:
        raise NotImplementedError
