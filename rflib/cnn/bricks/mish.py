# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:39:52 2020

@author: tycoer
"""
from .registry import ACTIVATION_LAYERS
import torch
@ACTIVATION_LAYERS.register_module()
class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()


    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x
