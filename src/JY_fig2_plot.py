import torch
import numpy as np
import os

random_seed = 6666
np.random.seed(random_seed)
torch.manual_seed(random_seed)

from model.Server import Server
from model.Client import Client

from utils import load_mnist, sample_iid, load_cifar, load_LFW
from conf import Args


args = Args()

