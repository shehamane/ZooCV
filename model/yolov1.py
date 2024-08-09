import torch
from torch import nn


class YoloV1(nn.Module):
    def __init__(self):
        super(YoloV1, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 192, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(1024, 1024, 3),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3),
            nn.LeakyReLU(),
        )
        self.region_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 20),
        )