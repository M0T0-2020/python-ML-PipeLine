import numpy as np
import pandas as pd

import torch
from torch import nn

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import transforms

from PIL import Image
import cv2


class ImageDataset_minist(Dataset):
    def __init__(self, data, data_shape, img_size):
        self.data = data
        self.data_shape = data_shape
        self.img_size = img_size
        self.transform = transforms.Compose(
            [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        
    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.data)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        # argumentation
        x = self.data[index]
        x = np.array(x)
        x = x.reshape(self.data_shape).astype(np.uint8)
        x = Image.fromarray(x, 'L')
        x = self.transform(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, img_path, img_size, base_path):
        self.img_path = img_path
        self.img_size = img_size
        self.base_path = base_path
        self.transform = transforms.Compose(
            [transforms.Resize(self.img_size), transforms.ToTensor()]
        )
        
    def read_img(self, path):
        img = cv2.imread(f"{self.base_path}{path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalize [0,1]
        img = cv2.normalize(img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return img

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.img_path)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        # argumentation
        path = self.img_path[index]
        img = self.read_img(path)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img