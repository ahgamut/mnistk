# -*- coding: utf-8 -*-
"""
    mnistk.run.optimizer
    ~~~~~~~~~~~~~~~~~~

    Handles all optimizer-related data

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import torch.optim as optim


def get_optimizer(**hypers):
    z = lambda params: optim.Adam(params, **hypers)
    return z
