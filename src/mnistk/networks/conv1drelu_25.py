# -*- coding: utf-8 -*-
"""
    conv1drelu_25.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dReLU_25(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=19, kernel_size=(38,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.ReLU(inplace=False)
        self.f2 = nn.Conv1d(in_channels=19, out_channels=37, kernel_size=(2,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Conv1d(in_channels=37, out_channels=55, kernel_size=(3,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f4 = nn.ReLU(inplace=False)
        self.f5 = nn.Conv1d(in_channels=55, out_channels=45, kernel_size=(8,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f6 = nn.ReLU(inplace=False)
        self.f7 = nn.Conv1d(in_channels=45, out_channels=63, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f8 = nn.ReLU(inplace=False)
        self.f9 = nn.Conv1d(in_channels=63, out_channels=10, kernel_size=(2,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=True, padding_mode='zeros')
        self.f10 = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        x = inputs[0]
        x = x.view(x.shape[0],16,49)
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
