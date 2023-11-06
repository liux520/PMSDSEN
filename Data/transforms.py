import torch
import random
import numpy as np
import torchvision.transforms
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numbers
import math
from torchvision.transforms import Pad
from torchvision.transforms import functional as F
from Metrics.psnr_ssim.matlab_functions import imresize

__all__ = ['RandomCrop', 'RandomCrop2',
           'RandomScale', 'RandomScale2', 'RandomResizedCrop', 'Resize',
           'RandomHorizontalFlip',
           'ColorJitter',
           'Compose',
           'MultiScale',
           'CenterCrop',
           'totensor', 'Normalize', 'Normalize2', 'ToTensor', ''
           ]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class Normalize(object):
    '''
        Normalize the tensors
    '''

    def __call__(self, sample):
        rgb_img, label_img = sample['image'], sample['label']
        rgb_img = F.to_tensor(rgb_img)  # convert to tensor (values between 0 and 1)
        rgb_img = F.normalize(rgb_img, mean, std)  # normalize the tensor
        label_img = torch.LongTensor(
            np.array(label_img).astype(np.int64))  # torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return {'image': rgb_img,
                'label': label_img}


class Normalize2(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=mean, std=std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class totensor(object):
    '''
    mean and std should be of the channel order 'rgb'
    '''

    def __init__(self, mean=mean, std=std):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['image'], im_lb['label']
        im, lb = np.array(im, dtype=np.uint8), np.array(lb, dtype=np.uint8)
        im = im.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
        im = torch.from_numpy(im).float().div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()  # .unsqueeze(0)
        return dict(image=im, label=lb)


class Resize(object):
    '''
        Resize the images
    '''

    def __init__(self, size=(512, 512)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, sample):
        rgb_img, label_img = sample['image'], sample['label']
        rgb_img = rgb_img.resize(self.size, Image.BILINEAR)
        label_img = label_img.resize(self.size, Image.NEAREST)
        return {'image': rgb_img,
                'label': label_img}


class RandomResizedCrop(object):
    '''
    Randomly crop the image and then resize it
    '''

    def __init__(self, size, scale=(0.5, 1.0), ignore_idx=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.ignore_idx = ignore_idx

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        rgb_img, label_img = sample['image'], sample['label']
        w, h = rgb_img.size

        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (
                math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        crop_size = (int(round(w * random_scale)), int(round(h * random_scale)))

        i, j, h, w = self.get_params(rgb_img, crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)

        rgb_img = rgb_img.resize(self.size, Image.ANTIALIAS)
        label_img = label_img.resize(self.size, Image.NEAREST)

        return {'image': rgb_img,
                'label': label_img}


class RandomCrop(object):
    def __init__(self, crop_size, pad=False, resize=True, ignore_idx=255, *args, **kwargs):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)  # size: (w, h)
        elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
            self.crop_size = crop_size

        self.resize = resize
        self.pad = pad
        self.ignore_idx = ignore_idx

    def __call__(self, sample):
        im = sample['image']
        lb = sample['label']
        assert im.size == lb.size
        w_crop, h_crop = self.crop_size
        w, h = im.size

        if (w_crop, h_crop) == (w, h):
            return dict(im=im, lb=lb)
        if w < w_crop or h < h_crop:
            # resize
            if self.resize:
                scale = float(w_crop) / w if w < h else float(h_crop) / h
                w, h = int(scale * w + 1), int(scale * h + 1)
                im = im.resize((w, h), Image.BILINEAR)
                lb = lb.resize((w, h), Image.NEAREST)
            # pad
            if self.pad:
                padw = (w_crop - w) // 2 + 1
                padh = (h_crop - h) // 2 + 1
                # TODO padding  fill to 0 , causing the network to learn street features on the black part of the image. ignore_index
                im = ImageOps.expand(im, (padw, padh, padw, padh), fill=0)  # (左，上，右，下)
                lb = ImageOps.expand(lb, (padw, padh, padw, padh), fill=self.ignore_idx)

        # sw, sh = random.random() * (w - w_crop), random.random() * (h - h_crop)
        sw, sh = random.randint(0, w - w_crop), random.randint(0, h - h_crop)
        crop = int(sw), int(sh), int(sw) + w_crop, int(sh) + h_crop
        return dict(
            image=im.crop(crop),
            label=lb.crop(crop)
        )


class RandomCrop2(object):
    '''
    Randomly crop the image
    '''

    def __init__(self, crop_size, ignore_idx=255):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
        self.ignore_idx = ignore_idx

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        rgb_img, label_img = sample['image'], sample['label']
        w, h = rgb_img.size
        pad_along_w = max(0, int((1 + self.crop_size[0] - w) / 2))
        pad_along_h = max(0, int((1 + self.crop_size[1] - h) / 2))
        # padd the images
        rgb_img = Pad(padding=(pad_along_w, pad_along_h), fill=0, padding_mode='constant')(rgb_img)
        label_img = Pad(padding=(pad_along_w, pad_along_h), fill=self.ignore_idx, padding_mode='constant')(label_img)

        i, j, h, w = self.get_params(rgb_img, self.crop_size)
        rgb_img = F.crop(rgb_img, i, j, h, w)
        label_img = F.crop(label_img, i, j, h, w)
        return {'image': rgb_img,
                'label': label_img}


class RandomScale(object):
    def __init__(self, scales=(0.75, 2), *args, **kwargs):
        self.scales = np.arange(scales[0], scales[1] + 0.1, 0.25)

    def __call__(self, sample):
        im = sample['image']
        lb = sample['label']
        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(round(W * scale)), int(round(H * scale))
        return dict(image=im.resize((w, h), Image.BILINEAR),
                    label=lb.resize((w, h), Image.NEAREST),
                    )


class RandomScale2(object):
    '''
    Random scale, where scale is logrithmic
    '''

    def __init__(self, scale=(0.5, 1.0)):
        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def __call__(self, sample):
        rgb_img, label_img = sample['image'], sample['label']
        w, h = rgb_img.size
        rand_log_scale = math.log(self.scale[0], 2) + random.random() * (
                    math.log(self.scale[1], 2) - math.log(self.scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        rgb_img = rgb_img.resize(new_size, Image.ANTIALIAS)
        label_img = label_img.resize(new_size, Image.NEAREST)
        return {'image': rgb_img,
                'label': label_img}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        im = sample['image']
        lb = sample['label']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(image=im,
                    label=lb,
                    )


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


class MultiScale(object):
    def __init__(self, scales=(1,)):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W * ratio), int(H * ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class CenterCrop(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)  # size: (w, h)
        elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
            self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        crop_w, crop_h = self.crop_size

        # center crop
        ww, hh = img.size
        w_center = ww // 2
        h_center = hh // 2
        half_w, half_h = crop_w // 2, crop_h // 2
        img_crop = img.crop((w_center - half_w, h_center - half_h, w_center + half_w, h_center + half_h))
        label_crop = label.crop((w_center - half_w, h_center - half_h, w_center + half_w, h_center + half_h))

        return {'image': img_crop,
                'label': label_crop}


###PyTorch 官方实现：
# from torchvision.transforms import functional as F
# from torch.nn import functional as F

if __name__ == '__main__':
    # img=Image.open('../cityscapes/train/aachen_000000_000019_leftImg8bit.png')
    # lb=Image.open('../cityscapes/train/aachen_000001_000019_leftImg8bit.png')
    # sample={'image':img,'label':lb}
    # out1=totensor()(sample)
    # out2=torchvision.transforms.ToTensor()(img)
    # print(torch.min(out1['image']),torch.max(out1['image']))
    # print(torch.min(out2),torch.max(out2))
    # print('---')

    x = np.ones((1, 3, 12, 12), dtype=np.int64)
    y = torch.from_numpy(x)
    z = torch.LongTensor(x)
    print(type(x), x.dtype, type(y), y.dtype, type(z), z.dtype)
