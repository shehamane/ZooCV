from data.base import Receiver, ClassificationDataset
import struct
from array import array
from os.path import join

import numpy as np


class MNISTReceiver(Receiver):
    train_images_filepath = 'raw/train-images-idx3-ubyte'
    train_labels_filepath = 'raw/train-labels-idx1-ubyte'
    test_images_filepath = 'raw/t10k-images-idx3-ubyte'
    test_labels_filepath = 'raw/t10k-labels-idx1-ubyte'

    def _load_data(self, images_rfilepath, labels_rfilepath):
        labels = []
        images_filepath = join(self.root_path, images_rfilepath)
        labels_filepath = join(self.root_path, labels_rfilepath)
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append(np.empty((28, 28)))
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i] = img

        return images, labels

    def __init__(self, root_path, train=True):
        self.root_path = root_path
        if train:
            self.X, self.y = self._load_data(self.train_images_filepath, self.train_labels_filepath)
        else:
            self.X, self.y = self._load_data(self.test_images_filepath, self.test_labels_filepath)

    def _get(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


class MNIST(ClassificationDataset):
    def __init__(self, root, transform=None, train=True):
        super().__init__(MNISTReceiver(root, train), transform)
        self.train = train

    def __len__(self):
        return len(self.receiver)
