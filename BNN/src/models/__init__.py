#%%======================================================
''''''
#%% Imports
from src.models.model_point.models_point import LinearNetwork, ConvNetwork_LeNet
from models_bbb import LinearNetwork as LinearNetworkBBB
from models_bbb import ConvNetwork_LeNet as ConvNetwork_LeNetBBB

#%%======================================================
''''''
#%% Model Factory
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model, input_size, output_size, q, args, training_mode=True):
        net = None

        if (model == "linear"):
            net = LinearNetwork(input_size, output_size, q, args)
        elif (model == "conv_lenet"):
            net = ConvNetwork_LeNet(input_size, output_size, q, args)
        elif (model == "linear_bbb"):
            net = LinearNetworkBBB(input_size, output_size, q, args)
        elif (model == "conv_lenet_bbb"):
            net = ConvNetwork_LeNetBBB(input_size, output_size, q, args)

        return net
