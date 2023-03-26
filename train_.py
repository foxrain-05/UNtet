import os
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet
from load_data import dataset

lr = 1e-3
batch_size = 4
epochs = 100

data_dir = Path('./data')
ckpt_dir = Path('./checkpoints')
log_dir = Path('./log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = dataset(data_dir="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = dataset(data_dir="val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)


for epoch in range(epochs + 1):
    for batch_idx, (_input, label) in enumerate(train_loader):
        
        output = net(_input)
        loss = fn_loss(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'Epoch: {epoch:4d}/{epochs} Batch: {batch_idx+1}/{len(train_loader)} loss: {loss.item():.6f}')

    with torch.no_grad():
        net.eval()

        for batch_idx, (_input, label) in enumerate(val_loader):
            output = net(_input)
            loss = fn_loss(output, label)

            print("it`s now validation time!")
            print(f"Epoch: {epoch:4d}/{epochs} Batch: {batch_idx+1}/{len(val_loader)} Loss: {loss.item():.6f}")
