import os
import shutil

import cv2
import torch
import torchvision
from torch.utils.data import Dataset


class Animals10(Dataset):
    ITLN2ENG = {'cane': 'dog', 'cavallo': 'horse', 'elefante': 'elephant', 'farfalla': 'butterfly', 'ragno': 'spider',
                'gallina': 'chicken', 'gatto': 'cat', 'mucca': 'cow', 'pecora': 'sheep', 'scoiattolo': 'squirrel', }
    NAME2LABEL = {
        'squirrel': 0,
        'dog': 1,
        'horse': 2,
        'elephant': 3,
        'spider': 4,
        'cow': 5,
        'butterfly': 6,
        'chicken': 7,
        'cat': 8,
        'sheep': 9,
    }

    NAMES = ('squirrel', 'dog', 'horse', 'elephant', 'spider', 'cow', 'butterfly', 'chicken', 'cat', 'sheep')

    def __prepare(self, src, dst):
        if not os.path.exists(dst):
            os.makedirs(dst)
        dst_images_dir = os.path.join(dst, 'images')
        if not os.path.exists(dst_images_dir):
            os.makedirs(dst_images_dir)
        dst_labels_path = os.path.join(dst, 'labels.txt')

        im_root_dir_path = os.path.join(src, 'raw-img')

        im_idx = 0

        with open(dst_labels_path, 'w') as labels_f:
            for class_name in os.listdir(im_root_dir_path):
                class_name_eng = self.ITLN2ENG[class_name]
                class_label = self.NAME2LABEL[class_name_eng]
                class_dir_path = os.path.join(im_root_dir_path, class_name)

                for im_name in os.listdir(class_dir_path):
                    im_path = os.path.join(class_dir_path, im_name)
                    dst_im_name = f'{im_idx}.jpeg'
                    dst_im_path = os.path.join(dst_images_dir, dst_im_name)

                    shutil.copy(im_path, dst_im_path)
                    labels_f.write(f'{dst_im_path} {class_label}\n')
                    im_idx += 1

        with open(dst_labels_path) as labels_f:
            self.labels = labels_f.read().splitlines()

    def __init__(self, src, dst, transform=None, prepare=False):
        src = os.path.abspath(src)
        dst = os.path.abspath(dst)

        if prepare:
            self.__prepare(src, dst)

        self.root_dir = dst

        if isinstance(transform, list) or isinstance(transform, tuple):
            self.transform = torchvision.transforms.Compose(transform)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_name = f'{idx}.jpeg'
        im_path = os.path.join(self.root_dir, 'images', im_name)
        im = cv2.imread(im_path)
        if self.transform:
            im = self.transform(im)

        label = int(self.labels[idx].strip().split(' ')[1])

        return im, label
