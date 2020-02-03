# -*- coding: utf-8 -*-
"""
    mnistk.tester
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
import numpy as np


class Tester(object):

    """Test the network on selected samples, and run other cpu-only tasks"""

    dataset = None

    def __init__(self):
        if Tester.dataset is None:
            Tester.dataset = datasets.MNIST(
                "data",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
        self.net = None

    def _load_network(self, net_name, weights):
        # Caller responsible for ensuring weights match
        net_cls = mnistk.networks.__dict__[net_name]
        self.net = net_cls().cpu().eval()
        self.net.load_state_dict(weights)

    def load_from_path(self, net_name, run_dir, epoch, weights_path=None):
        weights_path = weights_path or os.path.join(
            run_dir, "network-{}.pth".format(epoch)
        )
        assert net_name in weights_path, "Weights don't have the right net_name"
        weights = torch.load(weights_path, map_location=torch.device("cpu"),)
        self._load_network(net_name, weights)

    def predict_plus(self, img_tensor, true_val):
        nrm = lambda x: x.detach().cpu().numpy()
        img_tensor.requires_grad_(True)
        y_pred = self.net(img_tensor)
        backprops = []
        for i in range(10):
            img_tensor.grad = None
            y_pred[0, i].backward(retain_graph=True)
            backprops.append(nrm(img_tensor.grad)[0])

        imgs = {
            "input": nrm(img_tensor)[0],
            "grads": np.array(backprops),
        }
        scores = {
            "truth": true_val,
            "preds": nrm(torch.exp(y_pred)),
        }
        return imgs, scores

    def get_grads(self, i=None):
        if i is None:
            i = random.randint(0, 9999)
        return self.predict_plus(*self.dataset[i])

    @staticmethod
    def get_sample_grads(net_name, weights, i):
        t = Tester()
        t._load_network(net_name, weights)
        return t.get_grads(i)
