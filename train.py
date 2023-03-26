import os
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset 
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = Path('./data')
ckpt_dir = Path('./checkpoints')
log_dir = Path('./log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
