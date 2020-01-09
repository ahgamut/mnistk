# -*- coding: utf-8 -*-
"""
    resnetstyle_94.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

class ResNetStyle_94(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=43, kernel_size=(6, 6), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = BasicBlock(inplanes=43, planes=43)
        self.f2 = BasicBlock(inplanes=43, planes=43)
        self.f3 = BasicBlock(inplanes=43, planes=43)
        self.f4 = nn.Conv2d(in_channels=43, out_channels=32, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f5 = nn.Linear(in_features=9248, out_features=10, bias=False)
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = x.view(x.shape[0],9248)
        x = self.f5(x)
        x = self.f6(x)
        return x
