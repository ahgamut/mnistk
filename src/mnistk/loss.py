# -*- coding: utf-8 -*-
"""
    mnistk.loss
    ~~~~~~~~~~~~~

    Implements a LossFunc class that takes care of
    the loss computation

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import torch.nn as nn
import torch


class LossFunc(object):

    """Encapsulates the loss function computation, as the loss functions for ML can be detailed in their own right. Must be called with only two inputs and return an output that can be backpropagated."""

    def __init__(self):
        self.fn = nn.NLLLoss()

    def __call__(self, y_true, y_pred):
        loss = self.fn(y_pred, y_true)
        return loss
