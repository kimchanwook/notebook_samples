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
#%% Linear Network
class LinearNetwork(nn.Module):
    def __init__(self, input_size, output_size, q, args):
        super(LinearNetwork, self).__init__()
        self.args       = args
        self.input_size = 1

        for i in input_size:
            self.input_size *= int(i)
        self.output_size = int(output_size)

        #layers into a list
        layers = [100, 100, 100]
        self.layers = nn.ModuleList([])
        for i in range(len(layers)):
            if i==0:
                #Linear layers
                #    -bias=True: the layer will learn an additive bias
                self.layers.append(nn.Linear(self.input_size, int(layers[0]), bias=True))
            else:
                self.layers.append(nn.Linear(int(layers[i-1]), int(layers[i]), bias=True))

            #Relu: rectified linear activation function after each Linear layer: piecewise fn: f(x)=x for x positive.
            self.layers.append(nn.ReLU())

        #Stats...?
        #    -layers[len(layers)-1]: just last layer
        self.mu      = nn.Linear(int(layers[len(layers)-1]), 1, bias=True)
        self.log_var = nn.Linear(int(layers[len(layers)-1]), 1, bias=True)

        #quantisation
        self.q = q
        if self.q:
            #QuantStub: quantize stub module before calibration: same as an observer
            self.quant           = QuantStub()
            self.dequant_mu      = DeQuantStub()
            self.dequant_log_var = DeQuantStub()

    #Forward
    def forward(self, x):
        if (self.q):
            x = self.quant(x)
            x = clamp_activation(x, self.args)

        #Inserting inputs into layers
        for layer in self.layers:
            x = layer(x)
            x = clamp_activation(x, self.args)

        #Getting stats
        mu = self.mu(x)
        mu = clamp_activation(mu, self.args)
        log_var = self.log_var(x)
        log_var = clamp_activation(log_var, self.args)
        if self.q:
            mu      = self.dequant_mu(mu)
            log_var = self.dequant_log_var(log_var)

        #Return the last layer's outputs
        return(mu, log_var.exp())

    #Fusion is optional
    #    -Fuses a list of modules into a single module
    #    -But, it may save on memory access, make the run faster, and improve accuracy
    def fuse_model(self):
        fusion = []
        buf    = []
        #For each layers,
        for i, m in enumerate(self.layers):
            if ((isinstance(m, nn.Linear)) or (isinstance(m, nn.ReLU) and isinstance(self.layers[i-1], nn.Linear))):
                buf.append(str(i))
            if (len(buf) == 2):
                fusion.append(buf)
                buf = []
        # Cases
        # |\|\|
        # m=0, buff = ["0"]
        # m=1, buff = ["0", "1"]
        # m=2, buff = ["0", "1", "2"]
        # m=3, buff = ["0", "1", "2", "3"]
        # m=4, buff = ["0", "1", "2", 3", "4]
        #    -len(buf)=2: fusion=[[["0", "1", "2", "3", "4"]]]: buf=[]
        torch.quantization.fuse_modules(self.layers, fusion, inplace=True)


#%%======================================================
''''''
#%% Convolutional Linear Network
class ConvNetwork_LeNet(nn.Module):
    def __init__(self, input_size, output_size, q, args):
        super(ConvNetwork_LeNet, self).__init__()
        self.args = args
        self.init_channels = input_size[0]

        #Layers defined
        self.layers = nn.ModuleList([nn.Conv2d(in_channels=self.init_channels, out_channels=20, kernel_size=5, stride=1,
                                               padding=2, bias=False),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2, bias=False),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     Flatten(),
                                     nn.Linear(in_features=50*7*7, out_features=500, bias=False),
                                     nn.ReLU(),
                                     nn.Linear(in_features=500, out_features=output_size, bias=False)])

        #Quantization
        self.q = q
        if self.q:
            self.quant   = QuantStub()
            self.dequant = DeQuantStub()

    #Forward
    def forward(self, x):
        if self.q:
            x = self.quant(x)
            x = clamp_activation(x, self.args)

        # Passing inputs into each layers
        for layer in self.layers:
            x = layer(x)
            x = clamp_activation(x, self.args)

        if self.q:
            #Dequantization afterward
            x = self.dequant(x)

        #Softmax to normalize the output to a prob. distri.
        #    -dim=-1: take the last dimension
        x = F.softmax(x, dim=-1)

        return x

    #Fusion is optional
    #    -fuse only the second to last Linear + ReLu layer
    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, ["5", "6"], inplace=True)

#%%======================================================
''''''
#%% Residual Learning

#TODO: Residual learning for CIFAR Image classification
class BasicBlock(nn.Module):
    pass


class ConvNetwork_Resnet(nn.Module):
    pass

