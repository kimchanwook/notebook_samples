#%%======================================================
''''''
#%% Imports
import torch.nn.functional as F
import torch
import torch.nn as nn

#%% Constants
LOSS_FACTORY = {"classification": lambda args, scaling: ClassificationLoss(args, scaling),
                "regression": lambda  args, scaling: RegressionLoss(args, scaling)}

#%%======================================================
''''''
#%% LOSS CLASS
class Loss(nn.Module):
    def __init__(self, args, scaling):
        super(Loss, self).__init__()
        self.args    = args
        self.scaling = scaling

#%%======================================================
''''''
#%% Classification Loss
class ClassificationLoss(Loss):
    def __init__(self, args, scaling):
        super(ClassificationLoss, self).__init__(args, scaling)
        #Negative log likelihood loss
        self.ce = F.nll_loss()

    # Passing input data to the model executes the model's forward fn!
    #    -ex: pred=Model(x)
    #TODO: Check where the model is called with params (output, target, kl, ....)
    def forward(self, output, target, kl, gamma, n_batches, n_points):

        #loss_multiplier only used in sgld method: Stochastic Gradient Hamiltonian Monte Carlo
        if (self.scaling=="whole"):
            ce = n_points*self.ce(torch.log(output+1e-8), target) * self.args.loss_multiplier
            kl = kl / n_batches

        elif self.scaling=="batch":
            #Added 1e-8 to output to make sure non-zero?
            ce = self.ce(torch.log(output+1e-8), target)
            kl = kl / (target.shape[0]*n_batches)

        else:
            raise NotImplementedError("Other scaling not implemented!")

        #Total loss = cross entropy + gamma*kl
        #TODO: Check what gamma is
        loss = ce + gamma + kl

        return loss, ce, kl

#%%======================================================
''''''
#%% Regression Loss
class RegressionLoss(Loss):
    def __init__(self, args, scaling):
        super(RegressionLoss, self).__init__(args, scaling)

    def forward(self, output, target, kl, gamma, n_batches, n_points):

        #Constants
        mean      = output[0]
        var       = output[1]
        precision = 1/(var+1e-8)

        #For sgld method
        if (self.scaling=="whole"):
            heteroscedastic_loss = n_points * torch.mean(torch.sum(precision * (target - mean)**2 + \
                               torch.log(var+1e-8), 1), 0) * self.args.loss_multiplier
            kl = kl / n_batches

        elif (self.scaling=="batch"):
            heteroscedastic_loss = torch.mean(
                torch.sum(precision * (target - mean) ** 2 + torch.log(var + 1e-8), 1), 0)
            kl = kl / (target.shape[0] * n_batches)

        else:
            raise NotImplementedError("Other scaling not implemented!")

        #Total loss
        loss = heteroscedastic_loss + gamma*kl

        return loss, heteroscedastic_loss, kl













