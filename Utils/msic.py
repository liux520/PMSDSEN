import numpy as np
import torch
import random
from collections import OrderedDict
import time
import os
from natsort import os_sorted

# __all__=['set_seed', 'AverageMeter']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_state_dict(state_dict):
    "https://github.com/XU-GITHUB-curry/FBSNet/blob/main/utils/convert_state.py"
    """
    Converts a state dict saved from a dataParallel module to normal module state_dict inplace
    Args:
        state_dict is the loaded DataParallel model_state
    """
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove the prefix module.
        state_dict_new[name] = v
    return state_dict_new


def delete_state_module(weights):
    """
    From BasicSR
    """
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    # model.load_state_dict(weights_dict)
    return weights_dict


def weight_convert3(scale=2, origin_weights='', save_path=''):
    # new model
    from Model.PMSDSEN import PMSDSEN
    model_new = PMSDSEN(19, 'gelu')
    new_keys = [k for k in model_new.state_dict()]

    # old model
    from Model.PMSDSEN import PMSDSEN
    model_old = PMSDSEN(19, 'gelu')
    old_weights = torch.load(origin_weights, map_location='cpu')['state_dict']
    # for k, v in old_weights.items():
    #     print(f'{k}: {v.shape}')
    model_old.load_state_dict(old_weights)

    new_weights = {}
    for i, (k_old, v_old) in enumerate(old_weights.items()):
        new_weights[new_keys[i]] = v_old

    model_new.load_state_dict(new_weights)
    torch.save(new_weights,
               save_path)


def minmax_normalize(img, min_value=0, max_value=1):
    # range(0, 1)
    norm_img = (img - img.min()) / (img.max() - img.min())
    # range(min_value, max_value)
    norm_img = norm_img * (max_value - min_value) + min_value
    return norm_img


def meanstd_normalize(img, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    norm_img = (img - mean) / std
    return norm_img


def show_kv(weight_path):
    weights = torch.load(weight_path, map_location='cpu')
    for k, v in weights.items():  #['state_dict']
        if k == 'state_dict':
            pass
        else:
            print(k, v)
