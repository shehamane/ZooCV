import os.path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.metrics import Metric


class Logger(ABC):
    @abstractmethod
    def log_loss(self, idx, loss: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, idx, metrics: Dict[str, float]) -> None:
        raise NotImplementedError


class NotebookLogger(Logger):
    def __init__(self, metrics: List[Metric]):
        super().__init__(metrics)
        self.loss_dict = {'id': [], 'loss': []}
        self.metrics_dict = {'id': []}

    def log_loss(self, idx, loss: float):
        self.loss_dict['loss'].append(loss)
        print(f'Validation loss: {loss}')

    def log_metrics(self, idx, metrics: Dict[str, float]):
        self.metrics_dict['id'].append(idx)
        for name, val in metrics.items():
            if name not in self.metrics_dict.keys():
                self.metrics_dict[name] = []
            self.metrics_dict[name].append(val)

        print(f'Validation metrics: {metrics}')


    def get_log(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (pd.DataFrame(data=self.loss_dict).set_index('id'),
                pd.DataFrame(data=self.metrics_dict).set_index('id'))



class Printer(Logger):
    COLUMN_WIDTH = 10

    def log_loss(self, idx, loss: float):
        print(f'Loss for id {idx}: {loss:<6}')

    def log_metrics(self, idx, metrics: Dict[str, float]):
        print('=================')
        print(f'Metrics for id {idx}:')
        header = ' | '.join(f'{key:<{self.COLUMN_WIDTH}}' for key in metrics.keys())
        print(header)

        row = ' | '.join(f'{metrics[key]:<{self.COLUMN_WIDTH}}' for key in metrics.keys())
        print(row)
        print('=================')




import mlflow

class MLFlowLogger(Logger):
    def __init__(self, metrics: List[Metric], exp_name, host='127.0.0.1', port=8081):
        super().__init__(metrics)
        mlflow.set_tracking_uri(f'http://{host}:{port}')
        mlflow.set_experiment(exp_name)
        mlflow.start_run()

    def log_loss(self, idx, loss: float):
        mlflow.log_metric('loss', loss)

    def __del__(self):
        mlflow.end_run()
