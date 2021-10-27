# -*- coding: utf-8 -*-
"""
    mnistk.calibrator
    ~~~~~~~~~~~~~~

    Object to calibrate a network on samples of the QMNIST test set,
    and then obtain the 1-alpha confidence set via the RAPS method

    refer arxiv:2009.14193

    :copyright: (c) 2021 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import sys, os
import torch
from torch import nn
from torchvision import datasets, transforms
import mnistk.networks
from mnistk.run.utils import NDArrayEncoder
import random
import numpy as np
import json


class Calibrator(object):

    dataset = None
    dataloader = None
    batch_size = 10
    tuning_samples = 10000

    def __init__(self, alpha=0.1, k_reg=5, lamda=0.005):
        if Calibrator.dataset is None:
            Calibrator.dataset = datasets.QMNIST(
                "data",
                download=True,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
            Calibrator.dataloader = torch.utils.data.DataLoader(
                Calibrator.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
            )
        self.net = None
        self.gen_quantile = 0
        self.alpha = alpha
        self.k_reg = k_reg
        self.lamda = lamda

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
        weights = torch.load(
            weights_path,
            map_location=torch.device("cpu"),
        )
        self._load_network(net_name, weights)
        self._get_raw_scores()

    def _get_raw_scores(self):
        self._raw_scores = np.zeros((len(self.dataset), 10), np.float64)
        self._raw_labels = np.zeros(len(self.dataset), np.uint32)

        i = 0
        with torch.no_grad():
            for inp, label in self.dataloader:
                self._raw_scores[i : i + self.batch_size] = np.exp(
                    self.net(inp).cpu().detach().numpy()
                )
                self._raw_labels[i : i + self.batch_size] = label.cpu().detach().numpy()
                i += self.batch_size

        self._raw_indices = np.flip(self._raw_scores.argsort(axis=1), axis=1)
        self._raw_scores.sort(axis=1)
        self._raw_scores = np.flip(self._raw_scores, axis=1)

    def calibrate(self):
        # print("calibrating....")
        L = np.zeros(self.tuning_samples, dtype=np.uint32)
        E = np.zeros(self.tuning_samples, dtype=np.float32)
        rand = True
        scores = self._raw_scores[: len(L)]
        indices = self._raw_indices[: len(L)]
        labels = self._raw_labels[: len(L)]

        aa, L = (indices.T == labels).T.nonzero()
        for i in range(self.tuning_samples):
            E[i] = np.sum(scores[i, : L[i] + 1]) + self.lamda * max(
                0, L[i] - self.k_reg + 1
            )
        if rand:
            u = np.random.uniform(0, 1, len(L))
            E = E - scores[aa, L] + u * scores[aa, L]

        E.sort()
        self.gen_quantile = E[
            int(np.ceil((1 - self.alpha) * (1 + self.tuning_samples)))
        ]
        # print("generalized quantile is", self.gen_quantile)

    def get_C_single(self, index):
        scores = self._raw_scores[index]
        indices = self._raw_indices[index]
        label = self._raw_labels[index]
        L = 10
        for j in range(10):
            sc = np.sum(scores[0 : j + 1]) + self.lamda * max(0, L - self.k_reg)
            if sc > self.gen_quantile:
                L = j
                break
        if L > 0:
            V = (
                self.gen_quantile
                - np.sum(scores[0:L])
                - self.lamda * max(0, L - self.k_reg)
                + scores[L - 1]
            ) / scores[L - 1]
            U = np.random.uniform(0, 1, 1)[0]
            if V < U:
                L = L - 1
        return label, (indices[: L + 1], scores[: L + 1])

    def get_raps_full_info(self):
        set_sizes = np.zeros(len(self.dataset) - self.tuning_samples, dtype=np.float32)
        truth_index = (
            np.zeros(len(self.dataset) - self.tuning_samples, dtype=np.int32) - 1
        )
        for i in range(len(set_sizes)):
            actual_i = i + self.tuning_samples
            truth, (ind, scores) = self.get_C_single(actual_i)
            set_sizes[i] = len(ind)
            if truth in ind:
                truth_index[i] = (truth == ind).nonzero()[0][0]

        return {
            "alpha": float(self.alpha),
            "lamda": float(self.lamda),
            "k_reg": int(self.k_reg),
            "gen_quantile": float(self.gen_quantile),
            "set_sizes": set_sizes,
            "truth_index": truth_index,
        }

    @classmethod
    def get_raps_for_net(
        cls, net_name, run_name, epoch, alpha=0.05, k_reg=3, lamda=0.005
    ):
        """TODO: Docstring for get_raps_for_net.

        Args:
            net_index (TODO): TODO
            run_name  (TODO): TODO
            epoch     (TODO): TODO

        Kwargs:
            alpha (TODO): TODO
            k_reg (TODO): TODO
            lamda (TODO): TODO

        Returns: TODO

        """
        cal = Calibrator(alpha=alpha, lamda=lamda, k_reg=k_reg)
        cal.load_from_path(net_name, f"./results/{net_name}/runs/{run_name}/", epoch)
        cal.calibrate()
        info = cal.get_raps_full_info()
        info["run_name"] = run_name
        info["epoch"] = epoch
        return info

    def view_C_FGSM(self, index, epsmax=0.8):
        inp, label = self.dataset[index]
        zero, one = inp.min().item(), inp.max().item()
        inp.requires_grad_(True)
        loss = nn.NLLLoss()

        inps = []
        scs = []
        inps.append(inp)

        epsrange = np.arange(0, epsmax + 0.01, 0.01)
        for i, eps in enumerate(epsrange):
            img = inps[0]
            scores = self.net(img)
            cost = loss(scores, torch.LongTensor([label]))
            self.net.zero_grad()
            cost.backward()
            glitch = torch.clamp(inp + eps * inp.grad.sign(), zero, one)
            inps.append(glitch)

        for i, eps in enumerate(epsrange):
            with torch.no_grad():
                scores = self.net(inps[i])
            ind, scores = self.get_C_scores(scores)
            if len(ind) > 2 and i == 0:
                break
            elif len(ind) > 2:
                print(label, f"iteration {i}", eps, ind, scores)
                input()


def write_confidence_info(
    net_index, alpha=0.05, k_reg=3, lamda=0.005, dest_folder="./calibs"
):
    run_dict = {"t1": [1, 2, 3, 4], "t2": [2, 4, 6, 8], "t3": [4, 8, 12, 16]}
    net_name = mnistk.networks.NET_LIST[net_index].__name__
    answer = dict(name=net_name)
    info = []
    for k, v in run_dict.items():
        for val in v:
            print(net_index, net_name, f"run {k}", "epoch %02d" % (val), end="\r")
            info.append(
                Calibrator.get_raps_for_net(net_name, k, val, alpha, k_reg, lamda)
            )
    print()

    answer["info"] = info
    fname = os.path.join(dest_folder, f"{net_index}-{net_name}.json")
    with open(fname, "w") as f:
        json.dump(answer, f, cls=NDArrayEncoder, indent=4)
