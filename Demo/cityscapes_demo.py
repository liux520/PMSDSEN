import json
import os
import numpy as np
import scipy.misc as m
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn

### utils
from Metrics.psnr_ssim.matlab_functions import imresize

### transforms
from Data import transforms as transforms


def CityDataloader(train_batchsize=1,val_batchsize=1,crop_size=(1024,512),sr_scale=1,rescale=(0.75,2),**kwargs):
    train_set = CityscapesSegmentation(crop_size=crop_size,sr_scale=sr_scale,rescale=rescale,mode='train')
    val_set = CityscapesSegmentation(crop_size=crop_size,sr_scale=sr_scale,rescale=rescale,mode='val')
    test_set = CityscapesSegmentation(crop_size=crop_size,sr_scale=sr_scale,rescale=rescale,mode='test')
    num_class = train_set.NUM_CLASSES
    ignore_label = train_set.ignore_label
    train_loader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True, drop_last=True,**kwargs)#args.batch_size
    val_loader = DataLoader(val_set, batch_size=val_batchsize, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=val_batchsize, shuffle=False, **kwargs)
    #mean_r:72.43288125, mean_g:82.2808, mean_b:71.83093125, std_r:44.637559375, std_g45.827759375, std_b:44.94385625
    return train_loader, val_loader, test_loader, num_class, ignore_label



class CityscapesSegmentation(Dataset):
    NUM_CLASSES = 19
    def __init__(self,crop_size, sr_scale=2, rescale=(0.75,2), ignore_label=255 ,mode='train'):
        super(CityscapesSegmentation, self).__init__()
        assert mode in ('train', 'val', 'test')

        self.mode = mode
        self.ignore_label = ignore_label
        self.crop_size = crop_size #(1024,512)
        self.rescale = rescale
        self.sr_scale = sr_scale

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]   #16
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]   #19
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train','motorcycle', 'bicycle']

        self.class_map = dict(zip(self.valid_classes,range(self.NUM_CLASSES)))

        # rootpath
        rootpath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        json_path = os.path.join(rootpath, f'Data{os.sep}cityscapes_info.json')
        dataset_path = os.path.join(rootpath, f'Datasets{os.sep}Cityscapes')

        # json read
        with open(json_path, 'r') as f:
            labels_info = json.load(f)
        self.label_remap = {item['id']:item['trainId'] for item in labels_info}
        '''
        cityscapes ---  gtCoarse
                        gtFine      --- gtFine      --- train --- aachen  'aachen_000000_000019_gtFine_color.png'
                                                                          'aachen_000000_000019_gtFine_instanceIds.png'
                                                                          'aachen_000000_000019_gtFine_labelIds.png'
                                                        val
                                                        demo
                        leftImg8bit --- leftImg8bit --- train --- aachen  'aachen_000000_000019_leftImg8bit.png'
                                                        val
                                                        demo
        '''

        # Image:
        self.images= {}   #image_name:image_path
        self.images_name=[]
        mode_path=os.path.join(dataset_path,'leftImg8bit', mode)
        mode_folders=os.listdir(mode_path)
        for fd in mode_folders:
            fd_path=os.path.join(mode_path,fd)
            fd_imgs=os.listdir(fd_path)
            fd_img_names=[i.replace('_leftImg8bit.png','') for i in fd_imgs]
            fd_img_paths=[os.path.join(fd_path,i) for i in fd_imgs]
            self.images_name.extend(fd_img_names)
            self.images.update(dict(zip(fd_img_names,fd_img_paths)))
        # Label:
        self.labels = {}  # image_name:label_path
        self.label_name = []
        mode_path_ = os.path.join(dataset_path, 'gtFine', mode)
        mode_folders_ = os.listdir(mode_path_)
        for fd in mode_folders_:
            fd_path_ = os.path.join(mode_path_, fd)
            fd_imgs_ = os.listdir(fd_path_)
            fd_imgs_ = [i for i in fd_imgs_ if 'labelTrainIds' in i]   # labelTrainIds | labelIds
            fd_img_names_=[i.replace('_gtFine_labelTrainIds.png','') for i in fd_imgs_]  # labelTrainIds | labelIds
            fd_img_paths_ = [os.path.join(fd_path_, i) for i in fd_imgs_]
            self.label_name.extend(fd_img_names_)
            self.labels.update(dict(zip(fd_img_names_, fd_img_paths_)))

    def __getitem__(self, item):
        img_name = self.images_name[item]

        impth = self.images[img_name]
        lbpth = self.labels[img_name]
        image = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)

        # label_np = np.array(label,dtype=np.uint8)
        # label_np = self.convert_labels(label_np)
        # label_pil = Image.fromarray(label_np)   #label remap

        sample = {'image': image, 'label': label}
        if self.mode == 'train':
            return self.transform_tr(sample), img_name

        elif self.mode == 'val':
            return self.transform_val(sample), img_name

        elif self.mode == 'test':
            return self.transform_ts(sample), img_name


    def __len__(self):
        return len(self.images_name)

    def encode_segmap(self, label):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            label[label == _voidc] = self.ignore_label
        for _validc in self.valid_classes:
            label[label == _validc] = self.class_map[_validc]
        return label

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomScale2(self.rescale),
            transforms.RandomCrop2(self.crop_size,ignore_idx=255),
            transforms.RandomHorizontalFlip(),
            transforms.totensor(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # cutout(1,16)
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # transforms.CenterCrop(crop_size=self.crop_size),
            # transforms.Resize(size=self.crop_size),
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

    def convert_labels(self, label):
        for k, v in self.label_remap.items():
            label[label == k] = v
        return label

    def _get_paths_from_images(self,path):
        assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
        images = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    images.append(img_path)
        assert images, '{:s} has no valid image file'.format(path)
        return images

    def is_image_file(self,filename):
        IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



if __name__ == '__main__':
    # import os

    rootpath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    json_path = os.path.join(rootpath, f'Data/cityscapes_info.json')
    print(json_path)
    print(os.path.isfile(json_path))
    print(os.path.abspath(os.path.join(os.path.join(__file__, os.path.pardir, os.path.pardir), f'Datasets{os.sep}Cityscapes')))

    print(np.zeros((3,3)))