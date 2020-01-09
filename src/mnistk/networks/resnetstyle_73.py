# -*- coding: utf-8 -*-
"""
    resnetstyle_73.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

class ResNetStyle_73(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=56, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.Conv2d(in_channels=56, out_channels=10, kernel_size=(25, 25), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f2 = BasicBlock(inplanes=10, planes=10)
        self.f3 = BasicBlock(inplanes=10, planes=10)
        self.f4 = nn.Conv2d(in_channels=10, out_channels=59, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f5 = nn.Linear(in_features=59, out_features=10, bias=True)
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = x.view(x.shape[0],59)
        x = self.f5(x)
        x = self.f6(x)
        return x