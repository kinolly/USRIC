import os
import time

from pprint import pprint
from collections import namedtuple
from progress.bar import Bar, ShadyBar
from apex import amp
from PIL import Image

import utils.metrics as metrics
from utils.args import Args
from utils.utils import *
from model.unet import UNet

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim
import numpy as np 


class Trainer:
    def __init__(self, Args):
        self.args = Args
        self.start_epoch = 1
        self.epochs = self.args.epochs
        self.best_pred = 0
        self.best_miou = 0

        self.train_set = u_rsic.URSIC(mode='train')
        self.val_set = u_rsic.URSIC(mode='val')

        self.num_classes = self.train_set.NUM_CLASSES
        self.mean = self.train_set.mean
        self.std = self.train_set.std

        self.train_loader = DataLoader(self.train_set, batch_size=self.args.tr_batch_size,
                                       shuffle=True, num_workers=self.args.num_workers)
        self.val_loader = DataLoader(self.val_set, batch_size=self.args.vd_batch_size,
                                     shuffle=False, num_workers=self.args.num_workers)

        self.net = UNet(self.args.inplanes, self.num_classes).cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        if self.args.apex:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')
        self.net = nn.DataParallel(self.net, self.args.gpu_ids)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [40, 60], 0.3)

        self.Metric = namedtuple('Metric', 'pixacc miou')
        self.train_metric = self.Metric(pixacc=metrics.PixelAccuracy(ignore_index=0),
                                        miou=metrics.MeanIoU(self.num_classes))
        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(ignore_index=0),
                                        miou=metrics.MeanIoU(self.num_classes))

    def training(self, epoch):

        self.train_metric.miou.reset()
        self.train_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_train = len(self.train_loader)
        bar = Bar('Training', max=num_train)

        self.net.train()

        for idx, [img, tar] in enumerate(self.train_loader):
            if self.args.cuda:
                img, tar = img.cuda(), tar.cuda()

            self.optimizer.zero_grad()
            output = self.net(img)
            loss = self.criterion(output, tar)
            losses.update(loss.item())

            self.train_metric.pixacc.update(output, tar)
            self.train_metric.miou.update(output, tar)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f}'.format(
                batch=idx + 1,
                size=len(self.train_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                mIoU=self.train_metric.miou.get(),
                Acc=self.train_metric.pixacc.get(),
            )
            bar.next()
        bar.finish()
        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)
        if self.train_metric.pixacc.get() > self.best_pred and self.train_metric.miou.get() > self.best_miou:
            self.best_pred = self.train_metric.pixacc.get()
            self.best_miou = self.train_metric.miou.get()

            if self.train_metric.pixacc.get() > 0.5 and self.train_metric.miou.get() > 0.5:
                save_model(self.net, self.args.model_name, self.args.backbone, self.train_metric.pixacc.get(), self.train_metric.miou.get())


    def validation(self, epoch):

        self.val_metric.miou.reset()
        self.val_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_val = len(self.val_loader)
        bar = Bar('Validation', max=num_val)

        self.net.eval()

        for idx, [img, tar] in enumerate(self.val_loader):
            if self.args.cuda:
                img, tar = img.cuda(), tar.cuda()
            with torch.no_grad():
                output = self.net(img)
            loss = self.criterion(output, tar)
            losses.update(loss.item())

            self.val_metric.pixacc.update(output, tar)
            self.val_metric.miou.update(output, tar)
            # if epoch % 10 == 0:
            self.visualize_batch_image(img, tar, output, epoch, idx)
            
            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                mIoU=self.val_metric.miou.get(),
                Acc=self.val_metric.pixacc.get(),
            )
            bar.next()
        bar.finish()

        new_pred = self.val_metric.miou.get()
        metric_str = "Acc:{:.4f}, mIoU:{:.4f}".format(self.val_metric.pixacc.get(), new_pred)
        print('Validation:')
        print(f"[Epoch: {epoch}, numImages: {num_val * self.args.vd_batch_size}]")
        print(f'Valid Loss: {losses.avg:.4f}')

        return new_pred
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def visualize_batch_image(self, image, target, output, epoch, batch_index):
        # image (B,C,H,W) To (B,H,W,C)
        image_np = image.cpu().numpy()
        image_np = image_np.reshape([image_np.shape[0], image_np.shape[2], image_np.shape[3]])
        image_np *= self.std
        image_np += self.mean
        image_np *= 255.0
        image_np = image_np.astype(np.uint8)

        # target (B,H,W)
        target = target.cpu().numpy()

        # output (B,C,H,W) to (B,H,W)
        output = torch.argmax(output, dim=1).cpu().numpy()

        for i in range(min(3, image_np.shape[0])):
            img_tmp = image_np[i]
            target_tmp = decode_segmap(target[i]).astype(np.uint8)
            output_tmp = decode_segmap(output[i]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(131)
            plt.imshow(img_tmp, vmin=0, vmax=255)
            plt.subplot(132)
            plt.imshow(target_tmp, vmin=0, vmax=255)
            plt.subplot(133)
            plt.imshow(output_tmp, vmin=0, vmax=255)
            path = os.path.join('/home/arron/Documents/grey/vis_image', f'epoch_{epoch}')
            make_sure_path_exists(path)
            plt.savefig(f"{path}/{batch_index}-{i}.jpg")
            plt.close('all')
    
def train():
    args = Args()
    trainer = Trainer(args)

    print("==> Start training")
    print('Total Epoches:', trainer.epochs)
    print('Starting Epoch:', trainer.start_epoch)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)
        if not args.no_val:
            new_pred = trainer.validation(epoch)
            trainer.scheduler.step(new_pred)
            # trainer.auto_reset_learning_rate()

train()