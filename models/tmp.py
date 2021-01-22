import os
import time
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as T


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_path = 'unet_model.pth'

image_path = 'D:/code/project/datasets/train/images/'
mask_path = 'D:/code/project/datasets/train/masks/'

names = ['6caec01e67','2bfa664017','1544a0e952']
images = [Image.open(os.path.join(image_path, name+'.png')) for name in names]
masks = [Image.open(os.path.join(mask_path, name+'.png')) for name in names]

transforms = T.Compose([T.Grayscale(), T.ToTensor()])
x = torch.stack([transforms(image) for image in images])
y = torch.stack([transforms(mask) for mask in masks])

fig = plt.figure( figsize=(9, 9))

ax = fig.add_subplot(331)
plt.imshow(images[0])
ax = fig.add_subplot(332)
plt.imshow(masks[0])
ax = fig.add_subplot(333)
ax.imshow(x[0].squeeze(), cmap="Greys")
ax.imshow(y[0].squeeze(), alpha=0.5, cmap="Greens")

ax = fig.add_subplot(334)
plt.imshow(images[1])
ax = fig.add_subplot(335)
plt.imshow(masks[1])
ax = fig.add_subplot(336)
ax.imshow(x[1].squeeze(), cmap="Greys")
ax.imshow(y[1].squeeze(), alpha=0.5, cmap="Greens")

ax = fig.add_subplot(337)
plt.imshow(images[2])
ax = fig.add_subplot(338)
plt.imshow(masks[2])
ax = fig.add_subplot(339)
ax.imshow(x[2].squeeze(), cmap="Greys")
ax.imshow(y[2].squeeze(), alpha=0.5, cmap="Greens")

plt.show()