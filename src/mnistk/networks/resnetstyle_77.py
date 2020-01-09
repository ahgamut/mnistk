# -*- coding: utf-8 -*-
"""
    resnetstyle_77.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

class ResNetStyle_77(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(17, 17), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.f1 = BasicBlock(inplanes=40, planes=40)
        self.f2 = BasicBlock(inplanes=40, planes=40)
        self.f3 = BasicBlock(inplanes=40, planes=40)
        self.f4 = nn.Conv2d(in_channels=40, out_channels=42, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f5 = nn.Linear(in_features=6048, out_features=10, bias=False)
        self.f6 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = x.view(x.shape[0],6048)
        x = self.f5(x)
        x = self.f6(x)
        return x
