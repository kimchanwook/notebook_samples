#%%======================================================
''''''
#%% Imports
import sys
import argparse

from src.data import *
import src.utils as utils
from src.losses import LOSS_FACTORY
from src.models import ModelFactory

#%%Paths defined
sys.path.append("./data/")
sys.path.append("./src/")

#%% Argument parsing
#argparse obj to start parsing
parser = argparse.ArgumentParser("BNN_bbb_first")

#model type
parser.add_argument("--task", type=str, default="classification", help="the main task; defines loss")
parser.add_argument("--model", type=str, default="conv_lenet_bbb", help="the model that we want to train")

#Learning rate
parser.add_argument("--learning_rate", type=float, default=0.001, help="initial learning rate")
#Batch normalization
parser.add_argument("--loss_scaling", type=str, default="batch", help="smoothing factor")
#Weight decay
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
#Epochs
parser.add_argument("--epochs", type=int, default=100, help="num of training epochs")

#Data
parser.add_argument('--data', type=str, default="./data/", help="location of the data")
parser.add_argument("--dataset", type=str, default="mnist", help="dataset name")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--valid_portion", type=float, default=0.1, help="portion of training data")
parser.add_argument("--gamma", type=float, default=.1, help="portion of training data")
parser.add_argument("--sigma_prior", type=float, default=.1, help="portion of training data")

#Input/output sizes
#   -nargs=# of command line args
#       -"+" gathers all command-line ags into a list
parser.add_argument("--input_size", nargs="+", default=[1, 1, 28, 28], help="input size")
parser.add_argument("--output_size", type=int, default=10, help="output size")
parser.add_argument("--sameples", type=int, default=20, help="samples")

#Saving
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--save_last", action="store_true", default=True, help="whether to just save the last model")

#Status
parser.add_argument("--num_workers", type=int, default=16, help="number of workders")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--debug", action="store_true", help="whether we are currently debuggin")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device ids")

#Quantisation
parser.add_argument("--q", action="store_true", help="whether to do post training quantisation")
parser.add_argument("--at", action="store_true", help="whether to do training aware quantisation")

#%%======================================================
''''''
#%% Main class
def main():

    #Arguments to objects
    args = parser.parse_args()
    load = False

    #If conducting experiment, load set to True
    if args.save!="EXP":
        load=True

    #parse_args func defined in src/utils.py
    #TODO: Need to write parge_args in src/utils.py
    args, writer = utils.parse_args(args)

    #Start Re-training
    logging.info("#===Start Re-training")

    #Criterion for loss
    #    -returns loss, cross entropy, and KL divergence
    criterion = LOSS_FACTORY[args.task](args, args.loss_scaling)

    #In src/model_bbb/__init__.py
    #    -model_temp is a function
    model_temp = ModelFactory.get_model



#%%======================================================
''''''
#%%This is main
if __name__ == '__main__':
    main()