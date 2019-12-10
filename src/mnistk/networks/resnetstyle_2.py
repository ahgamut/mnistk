# -*- coding: utf-8 -*-
"""
    resnetstyle_2.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

class ResNetStyle_2(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=54, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = BasicBlock(inplanes=54, planes=54)
        self.f2 = nn.Conv2d(in_channels=54, out_channels=22, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Linear(in_features=8800, out_features=10, bias=False)
        self.f4 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = x.view(x.shape[0],8800)
        x = self.f3(x)
        x = self.f4(x)
        return x
