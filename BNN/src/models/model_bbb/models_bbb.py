#%%======================================================
''''''
#%% Imports
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.utils import Flatten, Add, clamp_activation
from torch.quantization import QuantStub, DeQuantStub

#%%======================================================
''''''
#%% Fuse BayesBackprob modules
def fuse_bbb_moduels(mod_list):
    pass

#%%======================================================
''''''
#%% Linear Network: Bayes-Bacprob
class LinearNetwork(nn.Module):
    pass

#%%======================================================
''''''
#%% Convolutional Linear Network: Bayes-Backprob
class ConvNetwork_LeNet(nn.Module):
    pass


#%%======================================================
''''''
#%% BasicBlock
class BasicBlock(nn.Module):
    pass

#%%======================================================
''''''
#%% ConvNetwork_Resnet
class ConvNetwork_Resnet(nn.Module):
    pass
