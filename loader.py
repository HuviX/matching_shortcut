from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset
import torch


class CustomLoader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.images_names = os.listdir(path)
        self.transform = transform
        self._class2name = dict()
        self._name2class = dict()


    def class2name(self, cl):
        return self._class2name[cl]

    
    def name2class(self, name):
        return self._name2class[name]
    

    def __len__(self):
        return len(self.images_names)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        self._class2name[idx] = self.images_names[idx]
        self._name2class[self.images_names[idx]] = idx


        img_name = os.path.join(self.path, self.images_names[idx])
        img = self.transform(img_name)
        return img
