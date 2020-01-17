# -*- coding: utf-8 -*-
"""
    toyprob.tester
    ~~~~~~~~~~~~~~

    Object to run a network on samples of the test set,
    and record other custom behaviour.

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import sys, os
import torch
from torchvision import datasets, transforms
import mnistk.networks
import random


class Tester(object):

    """Test the network on selected samples, and run other cpu-only tasks"""

    def __init__(self):
        self.samples = []
        self.dataset = datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.load_samples()
        self.net = None
        self.weight_dict = {}

    def load_samples(self):
        count = 0
        samples = {}
        for image, num in self.dataset:
            if samples.get(num, None) is None:
                samples[num] = (image.unsqueeze(0), num)
                count += 1
            if count == 10:
                break
        self.samples = [x[1] for x in sorted(samples.items(), key=lambda x: x[0])]

    def load_network(self, net_name, run_dir, epochs):
        self.net = mnistk.networks.__dict__[net_name]().cpu().eval()
        self.weight_dict = {}
        for epoch in epochs:
            self.weight_dict[epoch] = torch.load(
                os.path.join(run_dir, "network-{}.pth".format(epoch)),
                map_location=torch.device("cpu"),
            )

    def load_weights(self, epoch):
        weights = self.weight_dict[epoch]
        self.net.load_state_dict(weights)

    def predict_plus(self, img_tensor, true_val):
        nrm = lambda x: x.detach().cpu().numpy()
        img_tensor.requires_grad_(True)
        y_pred = self.net(x)[0]
        backprops = []
        for i in range(10):
            img_tensor.grad = None
            y_pred[i].backward(retain_graph=True)
            backprops.append(nrm(img_tensor.grad)[0, 0, :, :])

        return {
            "input": nrm(img_tensor[0, 0, :, :]),
            "truth": true_val,
            "preds": nrm(torch.exp(y_pred)),
            "backprops": backprops,
        }

    def get_class_sample(self, i):
        return self.predict_plus(*self.samples[i])

    def get_dataset_sample(self, i=None):
        if i is None:
            i = random.randint(0, 9999)
        return self.predict_plus(*self.dataset[i])
