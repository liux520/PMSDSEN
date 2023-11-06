import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
import cv2
from PIL import Image

from Demo.decode_segmap import decode_segmap
from Demo.utils import cityscapes_colorize_mask, camvid_colorize_mask


def check_dir(dir):
    for d in dir:
        if not os.path.exists(d):
            os.makedirs(d)


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process.
    cityscape uses different IDs for training and testing. So, change from Train IDs to actual IDs
    '''
    # img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def save_predict(pred, label, img_name, dataset, save_path, pred_labelid=False, pred_color=False, label_color=False):
    """
    :param pred:  numpy [h,w]
    :param label: numpy [h,w]
    :param img_name: aachen_000000_000019 | 0001TP_006690
    :param dataset:  cityscapes | camvid
    :param save_path:
    :param pred_labelid: Trainid -> Lableid for submitting offical website
    :param pred_color: network prediction output colorization
    :param label_color: GT label colorization
    :return:
    """
    check_dir([os.path.join(save_path, 'submit'),
               os.path.join(save_path, 'pred'),
               os.path.join(save_path, 'gt')])

    if pred_labelid:
        if dataset == 'cityscapes':
            output_labelid = relabel(pred.copy())
            output_labelid = Image.fromarray(output_labelid)#.resize((2048,1024),Image.BILINEAR)
            output_labelid.save(os.path.join(save_path, 'submit', img_name + '_gtFine_labelIds.png'))

    if pred_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(pred)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(pred)

        output_color.save(os.path.join(save_path, 'pred', img_name + '_pred.png'))

    if label_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(label)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(label)

        gt_color.save(os.path.join(save_path, 'gt', img_name + '_gt.png'))


class MscEvalV0(object):
    def __init__(self, device, dataset='cityscapes', scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale
        self.device = device
        self.dataset = dataset
        self.root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        print(f'scale: {scale}')

    @torch.no_grad()
    def __call__(self, net, dl, n_classes, save_results=False, val_or_test='val'):
        hist = torch.zeros((n_classes, n_classes), requires_grad=False).to(self.device)
        root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        if save_results:
            save_path = os.path.join(root, 'results', f'{self.dataset}{os.sep}{val_or_test}')
            if not os.path.exists(save_path): os.makedirs(save_path)

        for i, (sample, img_name) in enumerate(tqdm(dl)):
            imgs, label = sample['image'], sample['label']
            imgs, label = imgs.to(self.device), label.to(device)
            if i == 0:
                print(imgs.shape, label.shape)

            imgs50 = F.interpolate(imgs, scale_factor=self.scale, mode='bilinear', align_corners=True)
            logits = net(imgs50)[0]
            logits = F.interpolate(logits, scale_factor=1/self.scale, mode='bilinear', align_corners=True)

            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # [b,h,w]
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
            ).view(n_classes, n_classes).float()

            if save_results:
                out = preds.data.cpu().numpy().squeeze(0).astype(np.uint8)
                gt = label.data.cpu().numpy().squeeze(0).astype(np.uint8)
                save_predict(out, gt, img_name[0], self.dataset, save_path,
                             pred_labelid=True, pred_color=False, label_color=False)

        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        print(f'scale: {self.scale}, miou:{miou}')
        return miou.item()


if __name__ == "__main__":
    # 1. Basic Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 2. DataLoader
    from Demo.cityscapes_demo import CityDataloader
    from Demo.camvid_demo import CamvidDataloader

    city_tr, city_val, city_te, city_cls, city_ignore_label = CityDataloader(val_batchsize=1, crop_size=(1024, 512), )
    cam_tr, cam_val, cam_te, cam_cls, cam_ignore_label = CamvidDataloader(val_batchsize=1, crop_size=(1024, 512))

    # 3. Model
    from Demo.model_select import model_select
    model = model_select('PMSDSEN', cls=city_cls, act_name='gelu')
    model.to(device)
    model.eval()

    # 4. Eval
    print('Cityscapes Eval Start!')
    single_scale = MscEvalV0(device=device, dataset='cityscapes', scale=0.5, ignore_label=city_ignore_label)
    miou = single_scale(model, city_te, city_cls, save_results=True, val_or_test='test_city_75.02')

    # print('CamVid Eval Start!')
    # single_scale = MscEvalV0(device=device, dataset='camvid960', scale=0.5, ignore_label=cam_ignore_label)
    # miou = single_scale(model, cam_te, cam_cls, val_or_test='camvid960')  # 'val_7310'  'val960_bisenetv2'


"""
Val logging:
100.0000% weights from checkpoint is loaded!
100.0000% model params is init!
Drop keys:[]
PMSDSEN_ckpt_pred: 0.7502317451420611
scale: 0.5
torch.Size([1, 3, 1024, 2048]) torch.Size([1, 1024, 2048])
100%|██████████| 500/500 [01:58<00:00,  4.22it/s]
scale: 0.5, miou:0.7502354979515076

"""