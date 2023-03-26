import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

dir_data = Path('./data')

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(dir_data / name_label)
img_input = Image.open(dir_data / name_input)

nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = dir_data / "train"
dir_save_val = dir_data / 'val'
dir_save_test = dir_data / 'test'

dirs_path = [dir_save_train, dir_save_val, dir_save_test]
[os.makedirs(_dir, exist_ok=True) for _dir in dirs_path]
    
random_idx = np.random.permutation(np.arange(30))

datas_len = [nframe_train, nframe_val, nframe_test]
for i, dir_path in enumerate(dirs_path):
    index = random_idx[sum(datas_len[:i]):sum(datas_len[:i+1])]
    
    for j in index:
        img_label.seek(j)
        img_input.seek(j)

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(dir_path / f'label_{j:03d}.npy', label_)
        np.save(dir_path / f'input_{j:03d}.npy', input_)


if __name__ == '__main__':
    plt.subplot(121)
    plt.imshow(label_, cmap='gray')
    plt.title('Label')

    plt.subplot(122)
    plt.imshow(input_, cmap='gray')
    plt.title('Input')

    plt.show()