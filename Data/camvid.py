import json
import os
import numpy as np
import scipy.misc as m
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from natsort import os_sorted

### utils
from Metrics.psnr_ssim.matlab_functions import imresize

### transforms
from Data import transforms as transforms


def CamvidDataloader(train_batchsize=1, val_batchsize=1, crop_size=(960, 720), sr_scale=1, rescale=(0.75, 2), **kwargs):
    train_set = CamvidSegmentation(crop_size=crop_size, sr_scale=sr_scale, rescale=rescale, mode='trainval')
    val_set = CamvidSegmentation(crop_size=crop_size, sr_scale=sr_scale, rescale=rescale, mode='test')
    test_set = CamvidSegmentation(crop_size=crop_size, sr_scale=sr_scale, rescale=rescale, mode='test')
    num_class = train_set.NUM_CLASSES
    ignore_label = train_set.ignore_label
    train_loader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True, drop_last=True,
                              **kwargs)
    val_loader = DataLoader(val_set, batch_size=val_batchsize, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=val_batchsize, shuffle=False, **kwargs)
    # mean_r:72.43288125, mean_g:82.2808, mean_b:71.83093125, std_r:44.637559375, std_g45.827759375, std_b:44.94385625
    return train_loader, val_loader, test_loader, num_class, ignore_label


class CamvidSegmentation(Dataset):
    NUM_CLASSES = 11

    def __init__(self, crop_size, sr_scale=2, rescale=(0.75, 2), ignore_label=255, mode='train'):
        super(CamvidSegmentation, self).__init__()
        assert mode in ('train', 'val', 'test', 'trainval')

        self.mode = mode
        self.ignore_label = ignore_label
        self.crop_size = crop_size  # (1024,512)
        self.rescale = rescale
        self.sr_scale = sr_scale

        self.color_list = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                           [192, 192, 128], [64, 64, 128], [64, 64, 0],
                           [128, 64, 128], [0, 0, 192], [192, 128, 128],
                           [128, 128, 128], [128, 128, 0]]

        # rootpath
        rootpath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        dataset_path = os.path.join(rootpath,
                                    f'Datasets{os.sep}CamVid{os.sep}CamVid')  # Download from CamVid_PIDNet_archive

        # Image:
        if self.mode == 'train':
            self.images = [os.path.join(dataset_path, 'train', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'train')))]
            self.labels = [os.path.join(dataset_path, 'trainannot', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'trainannot')))]
        elif self.mode == 'val':
            self.images = [os.path.join(dataset_path, 'val', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'val')))]
            self.labels = [os.path.join(dataset_path, 'valannot', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'valannot')))]
        elif self.mode == 'test':
            self.images = [os.path.join(dataset_path, 'test', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'test')))]
            self.labels = [os.path.join(dataset_path, 'testannot', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'testannot')))]
        elif self.mode == 'trainval':
            self.images = [os.path.join(dataset_path, 'train', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'train')))] + \
                          [os.path.join(dataset_path, 'val', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'val')))]
            self.labels = [os.path.join(dataset_path, 'trainannot', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'trainannot')))] + \
                          [os.path.join(dataset_path, 'valannot', i) for i in
                           os_sorted(os.listdir(os.path.join(dataset_path, r'valannot')))]
            self.images, self.labels = os_sorted(self.images), os_sorted(self.labels)

    def __getitem__(self, item):
        impth = self.images[item]
        lbpth = self.labels[item]
        if os.path.basename(impth) != os.path.basename(lbpth):
            assert os.path.basename(impth) != os.path.basename(
                lbpth), f'{os.path.basename(impth)}, {os.path.basename(lbpth)} '
        image = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)  # color
        label = Image.fromarray(self.color2label(np.array(label)))  # 0-11 grey

        sample = {'image': image, 'label': label}
        if self.mode == 'train' or self.mode == 'trainval':
            return self.transform_tr(sample), os.path.splitext(os.path.basename(impth))[0]  # 0016E5_07959.png

        elif self.mode == 'val':
            return self.transform_val(sample), os.path.splitext(os.path.basename(impth))[0]

        elif self.mode == 'test':
            return self.transform_ts(sample), os.path.splitext(os.path.basename(impth))[0]

    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2]) * self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i
        return label

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]
        return color_map.astype(np.uint8)

    def __len__(self):
        return len(self.images)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomScale2(self.rescale),
            transforms.RandomCrop2(self.crop_size, ignore_idx=self.ignore_label),
            transforms.RandomHorizontalFlip(),
            transforms.totensor(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # transforms.CenterCrop(crop_size=self.crop_size),
            transforms.Resize(size=self.crop_size),
            transforms.totensor(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            # transforms.CenterCrop(crop_size=self.crop_size),
            # transforms.Resize(size=self.crop_size),
            transforms.totensor(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return composed_transforms(sample)

    def _get_paths_from_images(self, path):
        assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
        images = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    images.append(img_path)
        assert images, '{:s} has no valid image file'.format(path)
        return images

    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


if __name__ == '__main__':
    # import os

    rootpath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    json_path = os.path.join(rootpath, f'Data/cityscapes_info.json')
    print(json_path)
    print(os.path.isfile(json_path))
    print(os.path.abspath(
        os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), f'Datasets{os.sep}Cityscapes')))
