import numpy as np 
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from dataset import u_rsic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def make_dataset(tr_batch_size, vd_batch_size):
    train_set = u_rsic.URSIC(mode='train', batch_size=tr_batch_size)
    val_set = u_rsic.URSIC(mode='val', batch_size=vd_batch_size)
    return train_set, val_set, train_set.NUM_CLASSES

def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
    
class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 将输出转化为灰度图
def decode_segmap(output):
    output = output.astype(int)
    label_mask = np.zeros(output.shape, dtype=np.int16)

    label_mask[np.where(output == 0)] = 255
    label_mask[np.where(output == 1)] = 0
    return label_mask


# 将output拼接成完整的图片
class Merger:
    def __init__(self, ori_path, res_path, save_path):
        self.res_path = res_path
        self.save_path = save_path
        self.ori_list, self.height, self.width = get_ori_list_and_size(ori_path) 

    def merge_image(self):
        max_x, max_y = self.find_max_index(self.res_path, self.ori_list)
        for img_file in self.ori_list:
            ori_img_name = img_file.replace('.tif', '')
            res = np.zeros((self.height, self.width, 3))
            for x in range(max_x):
                for y in range(max_y):
                    img_name = '_'.join([ori_img_name, str(x), str(y)])
                    img_file = '.'.join([img_name, 'tif'])
                    img = np.array(Image.open(os.path.join(self.res_path, img_file)))
                    len_x, len_y, _ = img.shape
                    res[x * len_x:x * len_x + len_x, y * len_y:y * len_y + len_y, :] = img
            res_img = Image.fromarray(np.uint8(res))
            res_img.save(os.path.join(self.save_path, img_file))
            print(f"{ori_img_name} merge complete.")
    
    # 找出有多少张output可以组成一张原始图片
    def find_max_index(self):
        img_list = os.listdir(self.res_path)
        xs, ys = [], []

        for img_file in img_list:
            img_name, x, y = self.get_image_message(img_file)
            if self.ori_list[0].replace('.tif', '') == img_name[:-1]:
                xs.append(int(x))
                ys.append(int(y))
        return max(xs), max(ys)
            
    def get_image_message(self, img_file):
        split_tmp = img_file.split('_')[-2:]
        y, x = split_tmp[0], split_tmp[1].replace('.tif', '')
        return img_file.replace("_".join(split_tmp), ''), x, y

# 获得原始test图片的列表和尺寸
def get_ori_list_and_size(path):
    ori_list = os.listdir(path)
    height, width, _ = np.array(Image.open(os.path.join(path, ori_list[0]))).shape
    return ori_list, height, width