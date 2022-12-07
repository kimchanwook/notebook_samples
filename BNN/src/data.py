#%%======================================================
''''''
#%% Imports
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import random
import logging
import os
from os import path
from sklearn.model_selection import KFold
import zipfile
import urllib.request
from src.utils import BRIGHTNESS_LEVELS, SHIFT_LEVELS, ROTATION_LEVELS

#%% Constants
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
MNIST_MEAN, MNIST_STD = (0,), (1,)

#%%======================================================
''''''

#%% UCI data
class UCIDatasets():
    def __init__(self, name, data_path="", n_splits=10):
        self.name=name

    def _load_dataset(self):
        pass

#%% Transform data/image
class HorizontalTranslate(object):
    def __init__(self):
        pass
    def __call__(self, image):
        pass

#%% Reggression data generate
#Regresion function used for regression data generation
def regression_function(X, noise=True):
    pass

#Regression data generate
def regression_data_generator(N_points=100, X=None, noise=True):
    pass

#Load training data
def get_train_loaders(args, split=-1):

    #valid portion has to be in [0, 1]
    assert (args.valid_portion >=0 and args.valid_portion <1.)

    #Will store data here
    train_data = None

    if (args.dataset == "mnist"):
        pass




