import random
from re import I
from torch.utils.data import DataLoader, Subset
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataset import COVISDataset

# from pytorch
"""
The images are resized to resize_size=[256] using 
interpolation=InterpolationMode.BILINEAR, followed 
by a central crop of crop_size=[224]. Finally the 
values are first rescaled to [0.0, 1.0] and then 
normalized using mean=[0.485, 0.456, 0.406] and 
std=[0.229, 0.224, 0.225].
"""

def process_dataset(dataset, test_size = 0.2, batch_size = 32):
    
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




