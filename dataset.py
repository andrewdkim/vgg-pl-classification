from os import listdir
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class COVISDataset(Dataset):
    def __init__(self, dir):
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.dir = dir
        self.image_urls = listdir(dir)
        self.targets = [self.onehot_encode(x) for x in self.image_urls]
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