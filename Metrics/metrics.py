import numpy as np
import torch
import torch.distributed as dist

__all__ = ['Evaluator', 'Metrics']


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):  # todo: here if axis=0, origin value = 1
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU

    def Class_IoU(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        class_iou = dict(zip(range(self.num_class), IoU))
        return class_iou

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def compute_metrics(self):
        Acc = self.Pixel_Accuracy()
        Acc_class = self.Pixel_Accuracy_Class()
        miou = self.Mean_Intersection_over_Union()
        iou = self.Class_IoU()
        fwiou = self.Frequency_Weighted_Intersection_over_Union()
        metric_dict = dict(
            acc=Acc,
            acc_class=Acc_class,
            ious=iou,
            miou=miou.item(),
            fw_miou=fwiou.item(),
        )
        return metric_dict

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Metrics(object):
    def __init__(self, n_classes, lb_ignore=255, device='cpu'):
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.device = device
        self.confusion = torch.zeros((n_classes, n_classes), requires_grad=False).to(self.device).detach()

    @torch.no_grad()
    def reset(self):
        self.confusion = torch.zeros((self.n_classes, self.n_classes)).to(self.device).detach()

    @torch.no_grad()
    def update(self, label, preds):
        keep = (label >= 0) & (label < self.n_classes)  # label != self.lb_ignore
        preds, label = preds[keep], label[keep]
        self.confusion += torch.bincount(
            label * self.n_classes + preds,
            minlength=self.n_classes ** 2
        ).view(self.n_classes, self.n_classes)

    @torch.no_grad()
    def compute_metrics(self, ):
        # if dist.is_initialized():
        #     dist.all_reduce(self.confusion, dist.ReduceOp.SUM)

        confusion = self.confusion
        weights = confusion.sum(dim=1) / confusion.sum()
        tps = confusion.diag()
        # fps = confusion.sum(dim=0) - tps
        # fns = confusion.sum(dim=1) - tps

        # iou and fw miou
        ious = confusion.diag() / (confusion.sum(dim=0) + confusion.sum(dim=1) - confusion.diag())  # + 1
        #  ious = tps / (tps + fps + fns)   # + 1
        miou = ious.nanmean()
        fw_miou = torch.sum(weights * ious)

        # Acc, Acc class
        acc = tps.sum() / confusion.sum()
        acc_class = (tps / confusion.sum(dim=1)).nanmean()

        metric_dict = dict(
            acc=acc,
            acc_class=acc_class,
            ious=ious.tolist(),
            miou=miou.item(),
            fw_miou=fw_miou.item(),
        )

        # eps = 1e-6
        # # macro f1 score
        # macro_precision = tps / (tps + fps + 1)
        # macro_recall = tps / (tps + fns + 1)
        # f1_scores = (2 * macro_precision * macro_recall) / (
        #         macro_precision + macro_recall + eps)
        # macro_f1 = f1_scores.nanmean(dim=0)
        #
        # # micro f1 score
        # tps_ = tps.sum(dim=0)
        # fps_ = fps.sum(dim=0)
        # fns_ = fns.sum(dim=0)
        # micro_precision = tps_ / (tps_ + fps_ + 1)
        # micro_recall = tps_ / (tps_ + fns_ + 1)
        # micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + eps)
        #
        # # Acc, Acc class
        # acc = tps.sum() / confusion.sum()
        # acc_class = (tps / confusion.sum(dim=1)).nanmean()
        #
        # metric_dict = dict(
        #         weights=weights.tolist(),
        #         acc=acc,
        #         acc_class=acc_class,
        #         ious=ious.tolist(),
        #         miou=miou.item(),
        #         fw_miou=fw_miou.item(),
        #         f1_scores=f1_scores.tolist(),
        #         macro_f1=macro_f1.item(),
        #         micro_f1=micro_f1.item(),
        #         )
        return metric_dict


if __name__ == '__main__':
    from PIL import Image
    import json

    evaluate1 = Evaluator(19)
    evaluate2 = Metrics(19, )

    with open(r'D:\Software\Professional\AShare\Project\AServer\Seg\ASEG\Dataset\cityscapes_info.json') as f:
        labels_info = json.load(f)
    label_remap = {item['id']: item['trainId'] for item in labels_info}


    def convert_labels(label):
        for k, v in label_remap.items():
            label[label == k] = v
        return label


    lb = Image.open('../cityscapes/train_gt_fine/aachen_000000_000019_gtFine_labelIds.png')
    lb = np.array(lb)
    print(lb.max(), lb.min())
    lb_ = convert_labels(lb)
    lb_[lb_ == -1] = 44
    print(lb_.max(), lb_.min())
