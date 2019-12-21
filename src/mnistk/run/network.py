# -*- coding: utf-8 -*-
"""
    mnistk.run.network
    ~~~~~~~~~~~~~~~~

    Generates networks as defined by the requirements
    of the summary statistics problem.

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import torch.nn as nn
import torch

from mnistk.networks.basic import Basic
from mnistk.networks import NET_LIST


def construct(N=0):
    mod = NET_LIST[N]()
    name = mod.__class__.__name__
    if "Conv2d" in name or "ResNet" in name:
        mod.in_shape = [1, 28, 28]
    elif "Conv3d" in name:
        mod.in_shape = [1, 16, 7, 7]
    elif "Conv1d" in name:
        mod.in_shape = [16, 49]
    elif "Linear" in name:
        mod.in_shape = [784]
    else:
        mod.in_shape = [784]
    return mod
