import os

class Path:
    @staticmethod
    def db_root_dir(dataset_name):
        dataset_path_root = os.path.expanduser('~/dataset')
        if dataset_name == 'U-RSIC':
            return os.path.join(dataset_path_root, 'U-RSIC', 'complex')
        else:
            print('Dataset {} not available.'.format(dataset_name))
            raise NotImplementedError


class ProcessingPath:
    def __init__(self, img_type='img'):
        assert img_type == 'img' or img_type == 'label' or img_type == 'all'
        self.path_dict = {}
        self.img_type = img_type
        self.root_path = Path.db_root_dir('U-RSIC')

    def data_dir(self):
        if self.img_type == 'img':
            self.path_dict['img_type'] = 'img'
            self.path_dict['ori_path'] = os.path.join(self.root_path, 'train')

            self.path_dict['resize_path'] = os.path.join(self.root_path, 'train_2560x2560','img')
            self.path_dict['data_split_256'] = os.path.join(self.root_path, 'data_split_256x256','img')
            self.path_dict['aug_split_256'] = os.path.join(self.root_path, 'aug_split_1000x1000', 'img')
            self.path_dict['aug_split_1000'] = os.path.join(self.root_path, 'aug_split_1000x1000','img')

            self.path_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256x256','img')
            self.path_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256x256','img')
            self.path_dict['train_split_1000'] = os.path.join(self.root_path, 'train_split_1000x1000','img')
            self.path_dict['val_split_1000'] = os.path.join(self.root_path, 'val_split_1000x1000','img')

            self.path_dict['img_format'] = '.png'
        elif self.img_type == 'label':
            self.path_dict['img_type'] = 'label'
            self.path_dict['ori_path'] = os.path.join(self.root_path, 'complex_train_label')

            self.path_dict['resize_path'] = os.path.join(self.root_path, 'train_2560x2560','label')
            self.path_dict['data_split_256'] = os.path.join(self.root_path, 'data_split_256x256','label')
            self.path_dict['aug_split_256'] = os.path.join(self.root_path, 'aug_split_1000x1000', 'label')
            self.path_dict['aug_split_1000'] = os.path.join(self.root_path, 'aug_split_1000x1000','label')

            self.path_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256x256','label')
            self.path_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256x256','label')
            self.path_dict['train_split_1000'] = os.path.join(self.root_path, 'train_split_1000x1000','label')
            self.path_dict['val_split_1000'] = os.path.join(self.root_path, 'val_split_1000x1000','label')

            self.path_dict['img_format'] = '.tiff'
        else:
            self.path_dict['ori_path'] = os.path.join(self.root_path, 'complex_train_label')

            self.path_dict['resize_path'] = os.path.join(self.root_path, 'train_2560x2560')
            self.path_dict['data_split_256'] = os.path.join(self.root_path, 'data_split_256x256')
            self.path_dict['aug_split_256'] = os.path.join(self.root_path, 'aug_split_1000x1000')
            self.path_dict['aug_split_1000'] = os.path.join(self.root_path, 'aug_split_1000x1000')

            self.path_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256x256')
            self.path_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256x256')
            self.path_dict['train_split_1000'] = os.path.join(self.root_path, 'train_split_1000x1000')
            self.path_dict['val_split_1000'] = os.path.join(self.root_path, 'val_split_1000x1000')
            
            self.path_dict['image_format'] = '.png'
            self.path_dict['label_format'] = '.tiff'
        return self.path_dict