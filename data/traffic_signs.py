import os
from random import shuffle

import cv2
import torch
import torchvision
from torch.utils.data import Dataset


class TrafficSigns(Dataset):
    def _get_images_names(self):
        ims_dirpath = os.path.join(self.root_dirpath, 'images')
        names = list(os.listdir(ims_dirpath))
        return names

    def __init__(self, root, mode='train', transform=None):

        if mode not in ('train', 'test', 'val'):
            raise ValueError(f'Invalid mode: {mode}')
        self.mode = mode
        self.root_dirpath = os.path.join(root, mode)
        self.im_names = self._get_images_names()

        if isinstance(transform, list) or isinstance(transform, tuple):
            self.transform = torchvision.transforms.Compose(transform)
        else:
            self.transform = transform

    def __len__(self):
        labels_dirpath = os.path.join(self.root_dirpath, 'labels')
        return len(os.listdir(labels_dirpath))

    def __getitem__(self, idx):
        im_name = self.im_names[idx]
        im_path = os.path.join(self.root_dirpath, 'images', im_name)
        im = cv2.imread(im_path)
        if self.transform:
            im = self.transform(im)

        label_name = im_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.root_dirpath, 'labels', label_name)
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls_id, x, y, w, h = line.strip().split(' ')
                cls_id = int(cls_id)
                x, y, w, h = map(lambda n: float(n), [x, y, w, h])
                labels.append(cls_id)
                boxes.append([x, y, x + w, y + h])

        label = torch.Tensor(label)
        return im, label

