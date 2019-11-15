import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random

from PIL import Image
import albumentations as A

from .path import ProcessingPath
from .u_rsic_utils import *


class URSIC(Dataset):
    NUM_CLASSES = 2
    
    def __init__(self, mode='train'):
        assert mode in ['train', 'val']
        super().__init__()

        self._mode = mode

        self.path_obj = ProcessingPath('all')
        self._dir_dict = self.path_obj.data_dir()

        self.mean = 129.8762456088596
        self.std = 47.71913221750954

        if self._mode == 'train':
            self._image_dir = os.path.join(self._dir_dict['train_split_path'],'img')
            self._label_dir = os.path.join(self._dir_dict['train_split_path'],'label')

        if self._mode == 'val':
            self._image_dir = os.path.join(self._dir_dict['val_split_path'],'img')
            self._label_dir = os.path.join(self._dir_dict['val_split_path'],'label')

        self.img_format = self._dir_dict['image_format']
        self.label_format = self._dir_dict['label_format']
        self._img_list = os.listdir(self._image_dir)
        self._name_list = [name.split('.')[0] for name in self._img_list]
        self.len = len(self._name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
            return self.load_data(idx)

    def load_data(self, idx):
        img_path = os.path.join(self._image_dir, "".join([self._name_list[idx], self.img_format]))
        mask_path = os.path.join(self._label_dir, "".join([self._name_list[idx], self.label_format]))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = img.reshape(1, img.shape[0], img.shape[1])

        if self._mode == 'train':
            img = self.train_enhance(img)
        else:
            img = self.valid_enhance(img)
        mask = encode_segmap(mask)
        return img, mask.astype(np.long)

    def train_enhance(self, sample):
        compose = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.GaussNoise(),
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ])
        return compose(image=sample)['image']

    def valid_enhance(self, sample):
        compose = A.Compose([
            A.RGBShift(),
            A.InvertImg(),
            A.Blur(),
            A.GaussNoise(),
            A.Flip(),
            A.RandomRotate90(),
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ])
        return compose(image=sample)['image']

# if __name__ == '__main__':
#     trainset = URSIC('train')
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=8)
#     for img, label in train_loader:
#         print(img.shape, label.shape)
#         break