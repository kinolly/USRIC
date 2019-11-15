import os
import cv2
import math
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from progress.bar import Bar

from .path import ProcessingPath, Path

# 将原始图像resize为2560x2560
def img_resize(data_path, save_path, img_type, img_format, re_size=(2560, 2560)):
    img_list = os.listdir(data_path)
    num_imgs = len(img_list)
    bar = Bar(f'{img_type} Resizing:', max=num_imgs)

    i = 0
    for img_file in img_list:
        if img_file.split('.')[-1] != img_format.replace('.', ''):
            continue
            
        img = cv2.imread(os.path.join(data_path, img_file), cv2.IMREAD_GRAYSCALE)
        img_re = cv2.resize(img, re_size)
        cv2.imwrite(os.path.join(save_path, img_file), img_re)

        i += 1
        bar.suffix = f'{i}/{num_imgs}'
        bar.next()
    bar.finish()


# 将图像切割成若干256x256的小图片
class ImageSpliter:
    def __init__(self, path_dict, crop_size=(256, 256)):
        self.data_path = path_dict['data_path']
        self.save_path = path_dict['save_path']
        self.crop_size = crop_size
        self.data_list = []
        self.img_format = path_dict['img_format']

    def get_data_list(self):
        return self.data_list
        
    def split_image(self):
        img_list = os.listdir(self.data_path)
        num_imgs = len(img_list)

        for i, img_file in enumerate(img_list):
            img = np.load(os.path.join(self.data_path, img_file))
            self._img_crop(img, img_file, i)

    def _img_crop(self, img, img_file, i):
        _, height, width = img.shape
        len_y, len_x = self.crop_size
        img_name = img_file.replace(self.img_format, '')

        num_imgs = (height // len_y + 1) ** 2
        bar = Bar(f'Image {i + 1} spliting:', max=num_imgs)

        x = 0
        row_count = 0
        while x < height:
            y = 0
            col_count = 0
            while y < width:
                if y >= width - len_y and x >= height - len_x:
                    split_image = img[:, height - len_x:, width - len_y:]
                elif y >= width - len_y:
                    split_image = img[:, x:x + len_x, width - len_y:]
                elif x >= height - len_x:
                    split_image = img[:, height - len_x:, y:y + len_y]
                else:
                    split_image = img[:, x:x + len_x, y:y + len_y]

                split_image_name = '_'.join([img_name, str(row_count), str(col_count)])
                self.data_list.append(split_image_name)
                np.save(os.path.join(self.save_path, split_image_name), split_image)
                if y == width:
                    break

                y = min(width, y + len_y)
                col_count += 1
                bar.suffix = f'{row_count * (height // len_y + 1) + col_count}/{num_imgs}'
                bar.next()
        
            if x == height:
                break
            x = min(height, x + len_x)
            row_count += 1
        bar.finish()
        print('Image split all complete.')


# 切分训练集和验证集
def train_valid(data_list, paths_dict):
    _data_list = data_list.copy()
    num_names = len(_data_list)
    num_train = int(num_names * 0.9)
    num_val = num_names - num_train

    bar = Bar('Dividing:', max=num_names)
    
    for i in range(num_names):
        name = random.choice(_data_list)
        _data_list.remove(name)
        img_file = ''.join([name, paths_dict['image_format']])
        label_file = ''.join([name, paths_dict['label_format']])

        img_source = os.path.join(paths_dict['data_split_path'], 'img', img_file)
        label_source = os.path.join(paths_dict['data_split_path'], 'label', label_file)

        if i < num_train:
            img_target = os.path.join(paths_dict['train_split_path'], 'img')
            label_target = os.path.join(paths_dict['train_split_path'], 'label')
        else:
            img_target = os.path.join(paths_dict['val_split_path'], 'img')
            label_target = os.path.join(paths_dict['val_split_path'], 'label')
            
        shutil.copy(img_source, img_target)
        shutil.copy(label_source, label_target)
        bar.suffix = f'{i + 1}/{num_names}'
        bar.next()
    bar.finish()


# 将label灰度图转为segmap形式
def encode_segmap(label_img):
    label_img = label_img.astype(int)
    label_mask = np.zeros(label_img.shape, dtype=np.int16)
    label_opp = np.ones(label_img.shape, dtype=np.int16)

    label_mask[np.where(label_img == 255)] = 1
    label_mask = label_mask ^ label_opp
    
    return label_mask


# 将输出转化为灰度图
def decode_segmap(output):
    output = output.astype(int)
    label_mask = np.zeros(output.shape, dtype=np.int16)

    label_mask[np.where(output == 0)] = 255
    label_mask[np.where(output == 1)] = 0
    return label_mask


#  计算所有图片像素的均值并调用std
def mean_std(path):
    img_list = os.listdir(path)
    pixels_num = 0
    value_sum = 0
    files_num = len(img_list)
    bar = Bar('Calculating mean:', max=files_num)

    i = 0
    for img_file in img_list:
        img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
        pixels_num += img.size
        value_sum +=img.sum()
        i += 1
        bar.suffix = f'{i}/{files_num}'
        bar.next()
    bar.finish()

    value_mean = value_sum / pixels_num
    value_std = std(path, img_list, value_mean, pixels_num)
    return value_mean, value_std


# 计算所有图片的标准差
def std(path, img_list, mean, pixels_num):
    files_num = len(img_list)
    bar = Bar('Calculating std:', max=files_num)
    value_std = 0
    i = 0
    for img_file in img_list:
        img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
        value_std += ((img - mean) ** 2).sum()
        i += 1
        bar.suffix = f'{i}/{files_num}'
        bar.next()
    bar.finish()
    return math.sqrt(value_std / pixels_num)
        

# if __name__ == '__main__':
#     img_paths = ProcessingPath('img')
#     img_paths_dict = img_paths.data_dir()

#     label_paths = ProcessingPath('label')
#     label_paths_dict = label_paths.data_dir()

#     paths = ProcessingPath('all')
#     paths_dict = paths.data_dir()

#     img_resize(img_paths_dict['ori_path'], img_paths_dict['resize_path'], 
#                 img_paths_dict['img_type'], img_paths_dict['img_format'])
#     img_resize(label_paths_dict['ori_path'], label_paths_dict['resize_path'], 
#                 label_paths_dict['img_type'], label_paths_dict['img_format'])

#     img_spliter = ImageSpliter(img_paths_dict)
#     img_spliter.split_image()

#     label_spliter = ImageSpliter(label_paths_dict)
#     label_spliter.split_image()

#     data_name_list = img_spliter.get_data_list()

#     train_valid(data_name_list, paths_dict)

#     print(mean_std(img_paths_dict['train_split_path']))


