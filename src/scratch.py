# importing the libraries
import pandas as pd
import numpy as np
# for reading and displaying images
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
# PyTorch libraries and modules
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from . import config
from .dataset import CaptchaDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.41851586, 0.37706038, 0.52860934], 
        std=[0.25431615, 0.32462418, 0.08905637])
])

train_ds = CaptchaDataset(csv_file=config.TRAIN_DATA, root_dir=config.DATA_DIR, transform=transform)
train_loader = DataLoader(train_ds, batch_size=11, shuffle=True)

for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx == 0:
        imgs = [img for img in data.detach().numpy()]
        img_0 = imgs[0].reshape((200, 200, 3))
        print(img_0.shape)
        #imshow(img_0)
        print(target)
        #imshow(data)
        break

imshow(img_0)
plt.show()
img_1 = imgs[1].reshape((200, 200, 3))
imshow(img_1)
plt.show()