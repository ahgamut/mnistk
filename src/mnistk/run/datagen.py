# -*- coding: utf-8 -*-
"""
    mnistk.run.datagen
    ~~~~~~~~~~~~~~~~

    Contains the data generator class and helper methods.

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import numpy as np
import torch
from torchvision import datasets, transforms


class DataGen(object):
    def __init__(self, train_batch_size, test_batch_size):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=train_batch_size,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=test_batch_size,
            shuffle=False,
        )

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader

    def __str__(self):
        return "\n".join(
            ["DataGenerator", str(self.train_loader), str(self.test_loader)]
        )
