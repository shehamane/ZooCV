import os.path
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.metrics import Metric


class Logger(ABC):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    @abstractmethod
    def log_loss(self, idx, loss: float):
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, idx, metrics: Dict[Metric, float]):
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, name: str, artifact: Any):
        raise NotImplementedError


class NotebookLogger(Logger):
    def __init__(self, metrics: List[Metric]):
        super().__init__(metrics)
        self.metrics_dict = {metric: [] for metric in self.metrics} | {'id': [], 'loss': []}
        self.artifacts_dict = {}

    def log_loss(self, idx, loss: float):
        self.metrics_dict['loss'].append(loss)

    def log_metrics(self, idx, metrics: Dict[Metric, float]):
        self.metrics_dict['id'].append(idx)
        for metric, value in metrics.items():
            self.metrics_dict[metric].append(value)

    def log_artifact(self, name: str, artifact: Any):
        self.artifacts_dict[name] = artifact

    def get_log(self):
        return pd.DataFrame(data=self.metrics_dict).set_index(self.metrics_dict['id'], inplace=True)

    def get_artifacts_names(self):
        return list(self.artifacts_dict.keys())

    def get_artifact(self, name):
        return self.artifacts_dict[name]


class Printer(Logger):
    def __init__(self, metrics: List[Metric], save_path=None):
        super().__init__(metrics)
        self.print_header = True
        self.col_widths = {str(metric.name): len(metric.name) for metric in metrics}
        self.save_path = save_path

    def log_loss(self, idx, loss: float):
        print(f'Loss for id {idx}: {loss:<6}')

    def log_metrics(self, idx, metrics: Dict[str, float]):
        print('=================')
        print(f'Metrics for id {idx}:')
        if self.print_header:
            header = ' | '.join(f'{key:<{self.col_widths[key]}}' for key in self.col_widths.keys())
            print(header)
            self.print_header = False

        row = ' | '.join(f'{metrics[key]:<{self.col_widths[key]}}' for key in self.col_widths.keys())
        print(row)
        print('=================')


    def log_artifact(self, name: str, artifact: plt.Figure, ext='png'):
            if self.save_path is not None:
                plt.savefig(os.path.join(self.save_path, f'{name}.{ext}'))
            else:
                raise 'Save path is not specified'
