# -*- coding: utf-8 -*-
"""
    resnetstyle_44.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

class ResNetStyle_44(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=26, kernel_size=(19, 19), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = nn.ReLU(inplace=False)
        self.f2 = BasicBlock(inplanes=26, planes=26)
        self.f3 = nn.Conv2d(in_channels=26, out_channels=45, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f4 = nn.Linear(in_features=2205, out_features=10, bias=True)
        self.f5 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = x.view(x.shape[0],2205)
        x = self.f4(x)
        x = self.f5(x)
        return x
