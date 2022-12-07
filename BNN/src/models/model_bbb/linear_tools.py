#%%======================================================
''''''
#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch.autograd import Variable


#%%======================================================
''''''
#%% KL divergence defined
def kl_divergence(mu, sigma, mu_prior, sigma_prior):
    kl = 0.5 * (2 * torch.log(sigma_prior / sigma) - 1 + (sigma / sigma_prior).pow(2) + \
                ((mu_prior - mu) / sigma_prior).pow(2)).sum()
    return kl

#Softplus inverse
#    -softplus func is a approx. of ReLu activation func.
#    -often used in place of ReLu
def softplusinv(x):
    return torch.log(torch.exp(x)-1.)

#%%======================================================
''''''
#%% Linear: for LinearNetwork
class Linear(nn.Linear):
    r"""
    -Applies a linear transformation to the incoming data like torch.nn.Linear
    -Has additional obj. variables in addition to self.weight and self.bias
        -self.std_prior: prior information of a classifying param.
        -self.std: stddev
        -self.add_weight: function
        -self.mul_noise: function
    """

    def __init__(self, in_features, out_features, bias, sigma_prior=1.0, args=None):
        super(Linear, self).__init__(in_features, out_features, bias)

        #Parameter(): simple way to contain parameters of any model: subclass of Tensor
        #Weight: Obj attribute of nn.Linear: Uncertainty for each classifying parameter!!!
        #    -uniform_(): Fills self tensor with numbers sampled from the continuous uniform distri.
        #    -self.weight.data: data stored in "self.weight" parameter module.
        #        -filled the tensor with numbers samples from uniform distribution from (-0.01, 0.01)
        #Fill std: same dim as weight: Filled with -3
        #    -VERY SUBJECTIVELY DEFINED!
        #    -TODO: Check if any change to this range will alter final result
        self.std_prior = torch.nn.Parameter(torch.ones((1,))*sigma_prior, requires_grad=False)
        self.weight.data.uniform_(-0.01, 0.01)
        self.std = nn.Parameter(torch.zeros_like(self.weight).uniform_(-3, -3))

        #Operations defined:
        #    -nn.FloatFunctional(): State collector class for float operations
        #    -Can be used instad of calling "torch.": obj.add_weight.add(a, b): equivalent to torch.add(a, b)
        #Python grammar:
        #    -class object (FloatFunctional class) is assigned as an attribute of Linear object
        self.add_weight = torch.nn.quantized.FloatFunctional()
        self.mul_noise  = torch.nn.quantized.FloatFunctional()

        #Arguments
        self.args = args

        #If bias is given:
        if (self.bias is not None):
            self.bias.data.uniform_(-0.01, 0.01)

    #KL divergence
    #    -torch.zeros_like: return a tensor filled with zeros.
    #        -Return a tensor with same size as self.weight, but filled with zeroes.
    #    -to(): Performs Tensor dtype and/or device conversion.
    def get_kl_divergence(self):
        kl = kl_divergence(self.weight, F.softplus(self.std),
                           torch.zeros_like(self.weight).to(self.weight.device),
                           (torch.ones_like(self.std) * self.std_prior).to(self.weight.device))
        return kl

    #forward() defines how data will pass through the network.
    #    -Passing input data to the class model object "Linear" will execute the forward fn.
    def forward(self, x):
        output = None

        #self.training: an attribute of Module class (most basic)
        if (self.training):
            #mean
            #    -mm(): matric multiplication
            #    -weight.t(): transpose 2d matrix
            #    -transpose the weights (uncertainty for each classi. param!!!) and matrix multiply with the inpurts
            #        -Don't be consfused that "weights" defined here are the weights passed between layers in standard NN.
            #        -Ex: input: 1 data with 5 features: (1, 5)
            #            -weight (for each of 5 features): (1, 5): transposed -> (5, 1)
            mean = torch.mm(x, self.weight.t())

            #Stddev: VERY SUBJECTIVELY DEFINED!
            #    -\sqrt{1e-8 * x^2 * softplus(std^2)}
            #    -Square root of Input squared * std squared
            #    -TODO: Check if variation of this definition will alter the final result
            std = torch.sqrt(1e-8*torch.mm(torch.pow(x, 2), torch.pow(F.softplus(self.std).t(), 2)))

            #Add noise
            #    -torch.autograd.Variable
            #        -Variable around any object that will be trained and go through autograd and optimization.
            #    -TODO: Does this mean that only noise is being updated in training? Not mean or std? CHECK
            #    -torch.Tensor.normal(): fills tensor with samples from the normal distribution.
            #    -torch.Tensor.new(): returns a new tensor with size of mean.size()
            #        -torch.size(): return the size of the tensor
            noise = Variable(mean.new(mean.size()).normal_())

            #Bias
            bias = self.bias if (self.bias is not None) else 0.0

            #Final output
            output = mean + std*noise + bias

        else:
            #noise when not training, it is a variable that has the size of self.weight (uncertainty for class. params)
            #    -that is filled with random numbers samples from normal distri.
            #std: noise*std
            std           = F.softplus(self.std)
            noise         = Variable(self.weight.data.new(self.weight.size()).normal_())
            std           = self.mul_noise.mul(noise, std)
            weight_sample = self.add_weight.add(self.weight, std)
            bias          = self.bias if (self.bias is not None) else 0.0



        return output












#%%======================================================
''''''
#%% LinearReLU: for LinearNetwork
class LinearReLU(torch.nn.Sequential):
    pass


























