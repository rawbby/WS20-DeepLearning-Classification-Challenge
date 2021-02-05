# author: Robert Andreas Fritsch 1431348
# author: Jan Zoppe 1433409
# date: 2021-02-05

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class Network(nn.Module):
    view_size: int

    def __init__(self):
        """
        create a network for ChristmasImages
        """
        nn.Module.__init__(self)

        image_size_w = 256  # chosen / sync with data
        image_size_h = 256  # chosen / sync with data
        classes = 8  # sync with data

        kernel_conv1 = (4, 4)  # chosen
        kernel_conv2 = (4, 4)  # chosen
        kernel_conv3 = (4, 4)  # chosen

        i_conv1 = 3  # chosen
        o_conv1 = 11  # chosen

        i_conv2 = o_conv1
        o_conv2 = 15  # chosen

        i_conv3 = o_conv2
        o_conv3 = 15  # chosen

        pool_size = 2  # chosen

        self.view_size = o_conv3
        self.view_size *= int((((((image_size_w - kernel_conv1[0] + 1) / pool_size) - kernel_conv2[0] + 1) / pool_size) - kernel_conv3[0] + 1) / pool_size)
        self.view_size *= int((((((image_size_h - kernel_conv1[1] + 1) / pool_size) - kernel_conv2[1] + 1) / pool_size) - kernel_conv3[1] + 1) / pool_size)

        i_fc1 = self.view_size
        o_fc1 = int((i_fc1 - classes) * 0.4 + classes)  # generic

        i_fc2 = o_fc1
        o_fc2 = int((i_fc2 - classes) * 0.5 + classes)  # generic

        i_fc3 = o_fc2
        o_fc3 = classes

        self.conv1 = nn.Conv2d(i_conv1, o_conv1, kernel_size=kernel_conv1)
        self.conv2 = nn.Conv2d(i_conv2, o_conv2, kernel_size=kernel_conv2)
        self.conv3 = nn.Conv2d(i_conv3, o_conv3, kernel_size=kernel_conv3)
        self.pool = nn.MaxPool2d(pool_size, pool_size)

        self.drop = nn.Dropout(0.45)

        self.fc1 = nn.Linear(i_fc1, o_fc1)
        self.fc2 = nn.Linear(i_fc2, o_fc2)
        self.fc3 = nn.Linear(i_fc3, o_fc3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.drop(x)
        x = x.view(-1, self.view_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self):
        torch.save(self.state_dict(), 'model')

    @staticmethod
    def load_model():
        net = Network()
        net.load_state_dict(torch.load('model'))
        return net
