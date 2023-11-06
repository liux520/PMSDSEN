import os
import sys
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
# from dataset.mypath import Path
from matplotlib import pyplot as plt
from natsort import natsorted

__all__ = ['calculate_weigths_labels', 'calaulate_dataset_mean_std']


def calculate_weigths_labels(dataset, dataloader, num_classes, plot=False, save=False):
    """
    1. calculate_log_frequency
    Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.  However, the 1.1 is more frequently used!
    2. calculate_median_frequency
    Class balancing by median frequency balancing method.
    Reference: https://arxiv.org/pdf/1411.4734.pdf
       'a = median_freq / freq(c) where freq(c) is the number of pixels
        of class c divided by the total number of pixels in images where
        c is present, and median_freq is the median of these frequencies.'
    """

    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample, _ in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()

    if plot:
        class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                       'traffic light', 'traffic sign', 'vegetation', 'terrain',
                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                       'motorcycle', 'bicycle']
        plt.bar(class_names, z)
        plt.title("cityscape_train")
        plt.ylabel("Number")
        # plt.xlabel("class")
        plt.xticks(rotation=90, fontsize=13)
        for a, b in zip(class_names, z):
            plt.text(a, b + 1, b,
                     ha='center',
                     va='bottom',
                     rotation=90
                     )
        plt.tight_layout()
        plt.savefig('../dataset/city.png')

    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        # calc_log_frequency
        class_weight = 1 / (np.log(1.1 + (frequency / total_frequency)))  # 1.02
        # calc_median_frequency
        # class_weight = frequency / total_frequency
        # median_freq = (np.median(class_weight)) / class_weight

        class_weights.append(class_weight)
    ret = np.array(class_weights)
    if save:
        # classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
        classes_weights_path = './city_classes_weights.npy'
        np.save(classes_weights_path, ret)

    return ret, z


def calaulate_dataset_mean_std(train_path=None, val_path=None, test_path=None):
    mean = np.zeros(3, dtype=np.float32)  # [0.0,0.0,0.0]
    std = np.zeros(3, dtype=np.float32)

    # images_path = _get_paths_from_images(r'D:\Software\Professional\AShare\Project\AServer\Seg\ASEG\Datasets\Cityscapes\leftImg8bit\leftImg8bit')

    rootpath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    dataset_path = os.path.join(rootpath, f'Datasets{os.sep}Camvid')

    images_tr = [os.path.join(dataset_path, 'train', i) for i in
                 natsorted(os.listdir(os.path.join(dataset_path, r'train')))]
    images_val = [os.path.join(dataset_path, 'val', i) for i in
                  natsorted(os.listdir(os.path.join(dataset_path, r'val')))]
    images_test = [os.path.join(dataset_path, 'test', i) for i in
                   natsorted(os.listdir(os.path.join(dataset_path, r'test')))]
    images_path = images_tr + images_val  # + images_test

    len_imgs = len(images_path)
    for im in tqdm(images_path):
        img = Image.open(im)
        img_np = np.array(img, dtype=np.float32)
        mean[0] += np.mean(img_np[:, :, 0])
        mean[1] += np.mean(img_np[:, :, 1])
        mean[2] += np.mean(img_np[:, :, 2])

        std[0] += np.std(img_np[:, :, 0])
        std[1] += np.std(img_np[:, :, 1])
        std[2] += np.std(img_np[:, :, 2])

    print(f'images_len: {len_imgs}')
    print('未归一化')
    print(f'mean_r:{mean[0]}, mean_g:{mean[1]}, mean_b:{mean[2]}, '
          f'std_r:{std[0]}, std_g:{std[1]}, std_b:{std[2]}')
    print('归一化')
    print(f'mean_r:{mean[0] / len_imgs}, mean_g:{mean[1] / len_imgs}, mean_b:{mean[2] / len_imgs}, '
          f'std_r:{std[0] / len_imgs}, std_g:{std[1] / len_imgs}, std_b:{std[2] / len_imgs}')
    print('Tensor')
    print(
        f'mean_r:{mean[0] / len_imgs / 255.}, mean_g:{mean[1] / len_imgs / 255.}, mean_b:{mean[2] / len_imgs / 255.}, '
        f'std_r:{std[0] / len_imgs / 255.}, std_g:{std[1] / len_imgs / 255.}, std_b:{std[2] / len_imgs / 255.}')


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def enet_weighing(label, num_classes, c=1.02):
    class_count = 0
    total = 0

    label = label.cpu().numpy()

    # Flatten label
    flat_label = label.flatten()

    # Sum up the number of pixels of each class and the total pixel
    # counts for each label
    class_count += np.bincount(flat_label, minlength=num_classes)
    total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    class_weights = torch.from_numpy(class_weights).float()
    # print(class_weights)
    return class_weights


if __name__ == '__main__':
    # sys.path.append(r'/media/sr617/77C45040838A76BE/liux/AServer3/Segment')
    from Data.cityscapes import CityscapesSegmentation, CityDataloader
    from Data.camvid import CamvidSegmentation, CamvidDataloader

    # from Data.dataloader import make_data_loader

    opt = {}
    opt['crop_size'] = (480, 360)  # (2048,1024)
    opt['batch_size'] = 1
    opt['test_batch_size'] = 1
    opt['dataset'] = 'cityscapes'

    # dataset=CityscapesSegmentation(opt['batch_size'])
    # train_loader, val_loader, test_loader, num_class = CityDataloader(opt['batch_size'],opt['test_batch_size'],opt['crop_size'],2)
    #
    # out,per_pixel=calculate_weigths_labels(dataset,train_loader,num_class,plot=False,save=False)
    # print(out)
    # print(per_pixel)

    dataset = CamvidSegmentation(opt['crop_size'])
    train_loader, val_loader, test_loader, num_class = CamvidDataloader(opt['batch_size'], opt['test_batch_size'],
                                                                        opt['crop_size'], 1, )
    out, per_pixel = calculate_weigths_labels(dataset, train_loader, num_class, plot=False, save=False)
    print(out)
    print(per_pixel)
    #
    # calaulate_dataset_mean_std()


'''
!!! 1.1 !!!
[ 2.60119156  6.70640854  3.52139048  9.87654147  9.68428125  9.39731753
 10.28816395  9.96855555  4.34024419  9.45119163  7.62045441  9.40241026
 10.35877462  6.37420476 10.23334719 10.26747305 10.26344853 10.39409463
 10.09326082]
[2.02935879e+09 3.34577539e+08 1.25685226e+09 3.60609350e+07
 4.83128870e+07 6.75821510e+07 1.14443090e+07 3.03728540e+07
 8.75518320e+08 6.38706310e+07 2.21348839e+08 6.72293830e+07
 7.42758100e+06 3.84410250e+08 1.46026800e+07 1.26322990e+07
 1.28639550e+07 5.43984300e+06 2.28376040e+07]

'''

