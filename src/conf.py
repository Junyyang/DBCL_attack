"""
sample_rate: the proportion of clients selected in every communication rounds
number_client: the number of client
model_type: model to be trained
            CNN_sketch: CNN with sketch
            CNN: standard CNN
            NN: multilayer perceptron
            MLP_SketchLinear: multilayer perceptron with Sketch
attack: 0: not to attack
        1: perform attack

dataset: cifar/ mnist
dim_in: input dimension, only for mnist data
dim_out: dimension of output class, only for mnist data
p: parameter for random hashing in Sketch
round: communication rounds between Server and Clients
local_epochs: number of epochs for each client in one communication
local_batch_size: batch size for each client during local training
learningrate_client: learning rate for each client during local training
test_batch_size: batch size for test data
verbose: verbose setting for test
target: target accuracy for training to stop
gpu: 1 use gpu when available
     -1 use cpu

"""
import torch
import json

class Args:
    def __init__(self):
        config_file = './src/config.json'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     print("CUDA is available!")
        # else:
        #     print("CUDA is not available.")

        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.sample_rate = config['sample_rate']
        self.number_client = config['number_client']
        self.attack = config['attack']
        self.dim_in = config['dim_in']
        self.dim_out = config['dim_out']
        self.p = config['p']
        self.learningrate_server = config['learningrate_server']
        self.round = config['round']
        self.learningrate_client = config['learningrate_client']
        self.test_batch_size = config['test_batch_size']
        self.gpu = config['gpu']
        self.verbose = config['verbose']
        self.target = config['target']

        self.model_type = config['model_type']['0']
        self.datatype = config['datatype']['0']
        self.sketchtype = config['sketchtype']['0']

        if self.attack==1:
            self.local_epochs = 1
            self.local_batch_size = 1   # for attacking part
            self.p = 2                  # otherwise, s = math.floor(n / q) in sketch.py would be zero
            #self.number_client = 2
        else:
            self.local_epochs = 5
            self.local_batch_size = 50