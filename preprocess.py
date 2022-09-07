import random
from re import I
from torch.utils.data import DataLoader, Subset
import torch
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from os import listdir
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from pytorch
"""
The images are resized to resize_size=[256] using 
interpolation=InterpolationMode.BILINEAR, followed 
by a central crop of crop_size=[224]. Finally the 
values are first rescaled to [0.0, 1.0] and then 
normalized using mean=[0.485, 0.456, 0.406] and 
std=[0.229, 0.224, 0.225].
"""

def process_images(test_size = 0.2, batch_size = 32):
    data_dir = "data/RB"
    image_urls = listdir(data_dir)

    dataset = RBDataset(data_dir, image_urls)
    
    train_indices, test_indices, _, _ = train_test_split(
      range(len(dataset)), 
      dataset.targets,
      stratify=dataset.targets,
      test_size=test_size,
      random_state= random.randint(0, 100)
    )

    train_split = Subset(dataset, train_indices)
    test_split = Subset(dataset, test_indices)

    train_batches = DataLoader(train_split, batch_size = batch_size, shuffle=True, num_workers=4)
    test_batches = DataLoader(test_split, batch_size = batch_size, shuffle=False, num_workers=4)

    return train_batches, test_batches



class RBDataset(Dataset):
    def __init__(self, dir, image_urls):
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.dir = dir
        self.image_urls = image_urls
        self.targets = [self.onehot_encode(x) for x in image_urls]
        self.data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, index):
        image_url = self.image_urls[index]
        image = Image.open(self.dir + "/" + image_url)
        image_arr = np.array(image)
        image_arr = np.expand_dims(image_arr, -1)
        image_arr = image_arr.repeat(3, axis=-1)
        rgb_image = Image.fromarray(image_arr)
        image = self.data_transform(rgb_image)
        return image, self.targets[index]

    def onehot_encode(self, label):
        return 0 if label[0] == "A" else 1
