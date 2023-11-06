import cv2
from PIL import Image
import torch
import numpy as np
import os
from Utils.msic import _get_paths_from_images

cityscapes_trainIds2labelIds = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
                                        dtype=np.uint8)


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


def trainIDs2LabelID(trainID_png_dir, save_dir):
    print('save_dir:  ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    png_list = [im for im in os.listdir(trainID_png_dir) if '_gtFine_labelIds.png' in im]
    for index, png_filename in enumerate(png_list):
        png_path = os.path.join(trainID_png_dir, png_filename)
        print('Processing {}/{} ...'.format(index, len(png_list)))
        pngdata = np.array(Image.open(png_path))
        # trainID = pngdata  # model prediction
        # row, col = pngdata.shape
        # labelID = np.zeros((row, col), dtype=np.uint8)
        # for i in range(row):
        #     for j in range(col):
        #         labelID[i][j] = cityscapes_trainIds2labelIds[trainID[i][j]]

        # labelID = pngdata.copy()
        # valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]  # 19
        # class_map = dict(zip(range(19), valid_classes))
        # for c in range(19):
        #     labelID[labelID == c] = class_map[c]
        # labelID[labelID == 255] = 0

        labelID = pngdata.copy()
        labelID = relabel(labelID)

        res_path = os.path.join(save_dir, png_filename)
        new_im = Image.fromarray(labelID)
        new_im.save(res_path)


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=False, gt_color=False):
    if output_grey:
        output_grey = Image.fromarray(output)  # .resize((2048,1024),Image.BILINEAR)
        output_grey.save(os.path.join(save_path, img_name + '_gtFine_labelIds.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)

        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


cityscapes_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                      220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0,
                      70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

camvid_palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 60, 40, 222, 128, 128, 0, 192, 128, 128, 64,
                  64, 128, 64, 0, 128, 64, 64, 0, 0, 128, 192]

zero_pad = 256 * 3 - len(cityscapes_palette)
for i in range(zero_pad):
    cityscapes_palette.append(0)


# zero_pad = 256 * 3 - len(camvid_palette)
# for i in range(zero_pad):
#     camvid_palette.append(0)

def cityscapes_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cityscapes_palette)

    return new_mask


def camvid_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(camvid_palette)

    return new_mask


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


camvid_label_colors = np.array([
    [0, 0, 0],  # background
    [128, 0, 0],  # building
    [0, 128, 0],  # tree
    [128, 128, 0],  # sky
    [0, 0, 128],  # car
    [128, 0, 128],  # sign
    [0, 128, 128],  # pedestrian
    [128, 128, 128],  # fence
    [64, 0, 0],  # column_pole
    [192, 0, 0],  # road
    [64, 128, 0],  # sidewalk
    [192, 128, 0],  # bridge
    [64, 0, 128],  # river
    [192, 0, 128],  # tunnel
], dtype=np.uint8)


def camvid_color(seg_map):
    # 根据类别映射对分割结果进行着色
    seg_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for i in range(len(camvid_label_colors)):
        seg_image[seg_map == i] = camvid_label_colors[i]
    return seg_image


def processor(labelid_gts_path, preds_path, save_path):
    """ We use this function to process predict results for mask ignored class.
    Args:
        labelid_gts_path: Datasets/Cityscapes/gtFine/val
        preds_path: network output
        save_path: Save path of processed results

    Returns:

    """
    ignored_ids = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]).astype(np.uint8)
    color_map = [(0, 0, 0),
                 (0, 0, 0),
                 (0, 0, 0),
                 (0, 0, 0),
                 (0, 0, 0),
                 (111, 74, 0),
                 (81, 0, 81),
                 (128, 64, 128),
                 (244, 35, 232),
                 (250, 170, 160),
                 (230, 150, 140),
                 (70, 70, 70),
                 (102, 102, 156),
                 (190, 153, 153),
                 (180, 165, 180),
                 (150, 100, 100),
                 (150, 120, 90),
                 (153, 153, 153),
                 (153, 153, 153),
                 (250, 170, 30),
                 (220, 220, 0),
                 (107, 142, 35),
                 (152, 251, 152),
                 (70, 130, 180),
                 (220, 20, 60),
                 (255, 0, 0),
                 (0, 0, 142),
                 (0, 0, 70),
                 (0, 60, 100),
                 (0, 0, 90),
                 (0, 0, 110),
                 (0, 80, 100),
                 (0, 0, 230),
                 (119, 11, 32),
                 (0, 0, 142)]
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    id_set = np.array(a).astype(np.uint8)
    id_extra = np.array([-1]).astype(np.uint8)

    labelid_gts = _get_paths_from_images(labelid_gts_path, suffix='_labelIds.png')
    preds = _get_paths_from_images(preds_path)

    for lb, pd in zip(labelid_gts, preds):
        print(f'Processing: {os.path.basename(lb)}, {os.path.basename(pd)}')
        assert os.path.splitext(os.path.basename(lb))[0] == os.path.splitext(os.path.basename(pd))[0]

        sv_img = np.zeros((1024, 2048, 3)).astype(np.uint8)
        label = np.array(Image.open(lb).convert('P'))
        pred = np.array(Image.open(pd).convert('P'))

        for ig_id in ignored_ids:
            pred[label == ig_id] = ig_id
        for idx in id_set:
            for j in range(3):
                sv_img[:, :, j][pred == idx] = color_map[idx][j]
        for idx in id_extra:
            for j in range(3):
                sv_img[:, :, j][pred == idx] = color_map[34][j]

        sv_img = Image.fromarray(sv_img)
        sv_img.save(rf'{save_path}{os.sep}{os.path.basename(lb)}')


if __name__ == '__main__':
    import cv2
    import imageio
    from PIL import Image
    from PIL import ImageSequence

    # processor()
    img = Image.open(r"D:\Software\Professional\Github\Projects\PMSDSEN\Demo\video1_all.gif")
    i = 0
    for frame in ImageSequence.Iterator(img):
        frame.save(f"frame{i}d.png")
        i += 1
