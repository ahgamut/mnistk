# -*- coding: utf-8 -*-
"""
    mnistk.network
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
    return mod
