#%%======================================================
''''''
#%% Imports
import os
import numpy as np
import torch
import shutil
import random
import pickle
import torch.nn.functional as F
import sys
import time
import glob
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import re
import shutil
import uncertainty_metrics.numpy as um
import copy

#%% Constants
UINT_BOUNDS = {8: [0, 255], 7: [0, 127], 6: [0, 63], 5: [0, 31], 4: [0, 15], 3: [0, 7], 2: [0, 3]}
INT_BOUNDS = {8: [-128, 127], 7: [-64, 63], 6: [-32, 31],
                  5: [-16, 15], 4: [-8, 7], 3: [-4, 3], 2: [-2, 1]}
BRIGHTNESS_LEVELS = [(1.5, 1.5), (2., 2.), (2.5, 2.5), (3, 3), (3.5, 3.5)]
ROTATION_LEVELS = [(15,15),(30,30),(45,45),(60,60),(75,75)]
SHIFT_LEVELS = [0.1,0.2,0.3,0.4,0.5]

#%%======================================================
''''''

#%% Functions

def clamp_activation(x, args):
    #See torch quantization for quint8
    #    -args.activation_precision = one of [8, 6, 5, 4, 3, 2]
    #    -activation_precision declared in "experiments/run_all_quant.sh"
    #        -and passed as argument in "quantised" version of each model_bbb
    #torch.clamp = clamps all elements in inputs into the range [min, max]
    #    -remove any elements lying outside of [min, max]
    if x.dtype == torch.quint8:
        _min = (UINT_BOUNDS[args.activation_precision][0] - x.q_zero_point())*x.q_scale()
        _max = (UINT_BOUNDS[args.activation_precision][1] - x.q_zero_point())*x.q_scale()
        x    = torch.clamp(x, _min, _max)
    return x

def clamp_weight(x, args):
    if x.dtype == torch.quint8:
        _min = (UINT_BOUNDS[args.activation_precision][0] - x.q_zero_point())*x.q_scale()
        _max = (UINT_BOUNDS[args.activation_precision][1] - x.q_zero_point())*x.q_scale()
        x    = torch.clamp(x, _min, _max)
    return x

def parse_args(args, label=""):
    pass

#%%======================================================
''''''

#%% Classes

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        if len(x.shape)==1:
            #torch.unsqueeze = Return as new tensor with a dimension of size desired
            return x.unsqueeze(dim=0)
        #torch.reshape = change the shape of tensor
        #    -"-1"=single dimension
        return x.reshape(x.size(0), -1)

class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x, y):
        #torch.add = add two tensors
        return self.add.add(x, y)
















