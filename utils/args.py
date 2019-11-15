import torch
import os
class Args:

    def __init__(self):
        self.tr_batch_size = 64
        self.vd_batch_size = 32
        self.num_workers = 8
        self.inplanes = 1

        # self.model_name = 'unet'
        # self.backbone = 'resnet50'
        self.epochs = 100
        self.lr = 0.1
        self.no_val = False

        self.gpu_ids = [0, 1, 2]
        self.gpu_id = '0, 1, 2'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        # self.vis_image_dir = '/home/arron/Documents/grey/paper/vis_image/'
