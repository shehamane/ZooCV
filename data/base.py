import os
from abc import ABC
from typing import Tuple, Any

import torch
import cv2
from torch.utils.data import Dataset


class Receiver(ABC):
    def _get(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._get(idx)


class LocalReceiver(Receiver):
    def _load_labels(self):
        labels_path = os.path.join(self.root_path, 'labels.txt')
        with open(labels_path) as labels_f:
            self.labels = [line.strip().split(' ')[1] for line in labels_f.read().splitlines()]

    def __init__(self, root_path, dataset_type, ext='jpg'):
        self.root_path = root_path
        self.dataset_type = dataset_type

        if self.dataset_type == 'classification':
            self._load_labels()
            self.im_dir_path = os.path.join(root_path, 'images')

    def _get(self, idx: int) -> Tuple[torch.Tensor, Any]:
        im_name = f'{idx}.jpeg'
        im_path = os.path.join(self.im_dir_path, im_name)
        im = cv2.imread(im_path)
        label = self.labels[idx]

        return im, label

    def __len__(self):
        return len(self.labels)


class BaseDataset(Dataset):
    def __init__(self, receiver, transform=None):
        self.receiver = receiver
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


class ClassificationDataset(BaseDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        im, label = self.receiver[idx]
        if self.transform:
            im = self.transform(im)

        return im, label
