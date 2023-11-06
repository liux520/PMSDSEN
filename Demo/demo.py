import torch
import torch.nn as nn
import os
from natsort import os_sorted
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch.nn.functional as F


def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


city_label_colors = np.array([
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk 人行道
    [70, 70, 70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence  栅栏
    [153, 153, 153],  # pole  杆子
    [250, 170, 30],  # traffic light
    [220, 220, 0],  # traffic sign
    [107, 142, 35],  # vegetation  植被
    [152, 251, 152],  # terrain 地形
    [70, 130, 180],  # sky
    [220, 20, 60],  # person
    [255, 0, 0],  # rider  骑手
    [0, 0, 142],  # car
    [0, 0, 70],  # truck
    [0, 60, 100],  # bus
    [0, 80, 100],  # train
    [0, 0, 230],  # motorcycle
    [119, 11, 32],  # bicycle
], dtype=np.uint8)


def city_color(seg_map):
    # 根据类别映射对分割结果进行着色
    seg_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for i in range(len(city_label_colors)):
        seg_image[seg_map == i] = city_label_colors[i]
    return seg_image


if __name__ == '__main__':
    from Demo.model_select import model_select

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_select('PMSDSEN', cls=19, act_name='gelu')
    model.to(device)
    model.eval()

    path = r''
    save_path = r''
    imgs_list = _get_paths_from_images(path)

    with torch.no_grad():
        for im in tqdm(imgs_list):
            base, ext = os.path.splitext(os.path.basename(im))
            input_np = cv2.imread(im)[:, :, ::-1]
            input = uint2tensor(input_np).to(device)

            imgs50 = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)
            logits = model(imgs50)[0]
            logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)

            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # B, H, W
            preds = preds.data.cpu().numpy().squeeze(0).astype(np.uint8)
            seg_map = city_color(preds)

            display = np.concatenate([
                input_np, seg_map
            ], axis=1)

            cv2.imwrite(os.path.join(save_path, f'{base}{ext}'), display[:, :, ::-1])

