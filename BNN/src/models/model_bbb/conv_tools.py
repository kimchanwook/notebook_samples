#%%======================================================
''''''
#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy


#%%======================================================
''''''
#%% Convolution 2D: for ConvNetwork_LeNEt
class Conv2d(nn.Conv2d):
    pass

class ConvBn2d(torch.nn.Sequential):
    pass

class ConvReLU2d(torch.nn.Sequential):
    pass

class ConvBnReLU2d(torch.nn.Sequential):
    pass

def fuse_conv_bn_weight(conv_w, conv_b, conv_std, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    pass

def fuse_conv_bn_eval(conv, bn):
    pass

def fuse_conv_bn(conv, bn):
    pass

def fuse_conv_bn_relu(conv, bn, relu):
    pass
