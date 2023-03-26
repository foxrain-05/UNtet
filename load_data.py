import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from glob import glob


class dataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform

            self.label = glob(f"./data/{self.data_dir}/label*")
            self.input = glob(f'./data/{self.data_dir}/input*')

        def __len__(self):
            return len(self.label)

        def __getitem__(self, index):
            label = np.load(self.label[index])
            _input = np.load(self.input[index])

            if (label.ndim == 2) and (_input.ndim == 2):
                label = label[:, :, np.newaxis]
                _input = _input[:, :, np.newaxis]

            if self.transform:
                _input = self.transform(_input)
                label = self.transform(label)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            return _input.to(device), label.to(device)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
