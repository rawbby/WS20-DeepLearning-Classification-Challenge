# author: Robert Andreas Fritsch 1431348
# author: Jan Zoppe 1433409
# date: 2021-02-05

import torch
import torchvision
import torchvision.transforms as transforms

import os
import random

from PIL import Image
from natsort import natsorted


class ChristmasImages(torch.utils.data.Dataset):

    def __init__(self, path, training=True, categorized=None):
        torch.utils.data.Dataset.__init__(self)

        if categorized is None:
            categorized = training

        image_size_w = 256  # chosen / sync with network
        image_size_h = 256  # chosen / sync with network

        if training:
            self.transform = transforms.Compose(
                [transforms.Resize((image_size_w, image_size_h)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation(degrees=15.0),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((image_size_w, image_size_h)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.training = training
        self.categorized = categorized

        if categorized:
            self.data_set = torchvision.datasets.ImageFolder(root=path, transform=self.transform)

        else:
            self.data_set = os.listdir(path)

            if training:
                random.shuffle(self.data_set)
            else:
                self.data_set = natsorted(self.data_set)

            for i in range(len(self.data_set)):
                self.data_set[i] = os.path.join(path, self.data_set[i])

    def __getitem__(self, i):
        return self.data_set[i] if self.categorized else self.transform(Image.open(self.data_set[i]).convert("RGB"))

    def __len__(self):
        return len(self.data_set)

    def data_loader(self, batch_size=10):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=self.training, num_workers=4)
