# -*- coding: utf-8 -*-
"""
    conv2dtanh_24.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv2dTanh_24(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv2d(in_channels=1, out_channels=31, kernel_size=(10, 10), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Tanh()
        self.f2 = nn.Conv2d(in_channels=31, out_channels=49, kernel_size=(15, 15), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Tanh()
        self.f4 = nn.Conv2d(in_channels=49, out_channels=52, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f5 = nn.Tanh()
        self.f6 = nn.Conv2d(in_channels=52, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f7 = nn.Tanh()
        self.f8 = nn.Conv2d(in_channels=32, out_channels=45, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f9 = nn.Conv2d(in_channels=45, out_channels=10, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.f10 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],1,28,28)
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        x = self.f9(x)
        x = x.view(x.shape[0],10)
        x = self.f10(x)
        return x
