import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import argparse
import os
import random
from imgclass import Images







if __name__ == "__main__" :
    paraser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        default = './Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset',
                        help = 'directory of image files')

    args = parser.parse_args()
