# -*- coding: utf-8 -*-
"""
    resnetstyle_60.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

class ResNetStyle_60(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = BasicBlock(inplanes=10, planes=10)
        self.f2 = nn.Conv2d(in_channels=10, out_channels=17, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.ReLU(inplace=False)
        self.f4 = nn.Linear(in_features=8228, out_features=10, bias=False)
        self.f5 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = x.view(x.shape[0],8228)
        x = self.f4(x)
        x = self.f5(x)
        return x
