# -*- coding: utf-8 -*-
"""
    resnetstyle_88.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

class ResNetStyle_88(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=53, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = BasicBlock(inplanes=53, planes=53)
        self.f2 = nn.Conv2d(in_channels=53, out_channels=59, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f3 = nn.ReLU(inplace=False)
        self.f4 = nn.Linear(in_features=39884, out_features=117, bias=False)
        self.f5 = nn.ReLU(inplace=False)
        self.f6 = nn.Linear(in_features=117, out_features=10, bias=False)
        self.f7 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = x.view(x.shape[0],39884)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x
