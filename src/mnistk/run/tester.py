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
from mnistk.run import DataGen, construct
from mnistk.networks import NET_LIST


class Tester(object):

    """Test the network on selected samples, and run other cpu-only tasks"""

    def __init__(self, net_id, run, result_dir, dest, net=None):
        if net is None:
            assert isinstance(net, NET_LIST[net_id])
            self.net = net.cpu()
        else:
            self.net = construct(net_id).cpu()
        self.run_name = run
        self.result_dir = result_dir
        self.dest = dest
        self.samples = []

    def class_samples(self):
        count = 0
        samples = {}
        test = self.datagen.test()
        for xt, yt in test:
            for i, y in enumerate(yt):
                if samples.get(y.item(), None) is None:
                    samples[y.item()] = (y, xt[i])
                    count += 1
            if count == 10:
                break
        self.samples = sorted(samples.items(), key=lambda x: x[0])
        self.samples = [x[1] for x in self.samples]

    def get_backprops(self):
        inps = []
        outs = []
        for yp, xp in self.samples:
            x, y_true = (
                xp.unsqueeze(0).to(self.device),
                yp.unsqueeze(0).to(self.device),
            )
            x.requires_grad_(True)
            y_pred = self.net(x)
            inps.append(x)
            outs.append(y_pred)
        return inps, outs
