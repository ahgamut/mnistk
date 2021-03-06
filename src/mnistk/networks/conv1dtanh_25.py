# -*- coding: utf-8 -*-
"""
    conv1dtanh_25.py

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: MIT
"""
import torch
from torch import nn

class Conv1dTanh_25(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.f0 = nn.Conv1d(in_channels=16, out_channels=52, kernel_size=(13,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f1 = nn.Tanh()
        self.f2 = nn.Conv1d(in_channels=52, out_channels=52, kernel_size=(35,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f3 = nn.Tanh()
        self.f4 = nn.Conv1d(in_channels=52, out_channels=58, kernel_size=(2,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f5 = nn.Tanh()
        self.f6 = nn.Conv1d(in_channels=58, out_channels=37, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f7 = nn.Tanh()
        self.f8 = nn.Conv1d(in_channels=37, out_channels=41, kernel_size=(1,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f9 = nn.Tanh()
        self.f10 = nn.Conv1d(in_channels=41, out_channels=10, kernel_size=(2,), stride=(1,), padding=(0,), dilation=(1,), groups=1, bias=False, padding_mode='zeros')
        self.f11 = nn.LogSoftmax(dim=1)

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
        x = self.f10(x)
        x = x.view(x.shape[0],10)
        x = self.f11(x)
        return x
