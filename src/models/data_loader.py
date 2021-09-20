from __future__ import print_function, division
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from skimage import io, transform

from .. import config


class CaptchaDataset(Dataset):
    """Captcha image dataset
    
        Inspired by: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.squares_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.squares_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.squares_frame.iloc[idx, 0])

        image = io.imread(img_name)
        squares = self.squares_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, squares



if __name__ == "__main__":
    """Call this script for viewing a couple of the images"""
    import matplotlib.pyplot as plt
    captcha_dataset = CaptchaDataset(csv_file=config.TRAIN_DATA,
                                     root_dir=config.PROC_DIR)

    fig = plt.figure()

    for i in range(4, 4+len(captcha_dataset)):
        data, target = captcha_dataset[i]

        print(i, data.shape, target)

        ax = plt.subplot(1, 4, i - 3)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(data)

        if i == 7:
            plt.show()
            break
