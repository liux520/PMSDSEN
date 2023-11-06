import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torch.cuda.amp as amp


class OhemCrossEntropy(nn.Module):
    """
    https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py
    """
    def __init__(self, ignore_label=-1, thres=0.7, weight=None,
                 n_img_per_gpu=6, cropsize=(1024, 512), min_kept=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = n_img_per_gpu * cropsize[0] * cropsize[1] // 16  # max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = [0.4, 1.0]
        sb_weights = 1.0
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])

        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)

        else:
            raise ValueError("lengths of prediction and target are not identical!")


class SegmentationLoss(nn.Module):
    def __init__(self, n_classes=21, loss_type='ce', device='cuda', ignore_idx=255, class_wts=None):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.device = device
        self.ignore_idx = ignore_idx
        self.smooth = 1e-6
        self.class_wts = class_wts.to(device) if class_wts is not None else class_wts

        if self.loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_wts)
        elif self.loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=self.class_wts)
        elif self.loss_type == 'focal':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=self.class_wts)
        elif self.loss_type == 'focalv2':
            self.loss_fn = FocalLoss(alpha=0.25, gamma=2)

    def CrossEntropyLoss(self, inputs, target):
        if isinstance(inputs, (tuple, list)):
            tuple_len = len(inputs)
            # assert tuple_len == 2
            loss = 0
            for i in range(tuple_len):
                loss += self.loss_fn(inputs[i], target)
        else:
            loss = self.loss_fn(inputs, target)
        return loss

    def FocalLoss_v1(self, inputs, target, gamma=2, alpha=0.5):
        if isinstance(inputs, (tuple, list)):
            tuple_len = len(inputs)
            # assert tuple_len == 2
            loss = 0
            for i in range(tuple_len):
                n, c, h, w = inputs.size()
                logpt = -self.loss_fn(inputs, target)  # .long()
                pt = torch.exp(logpt)
                if alpha is not None:
                    logpt *= alpha
                loss_temp = -((1 - pt) ** gamma) * logpt
                loss += loss_temp
        else:
            n, c, h, w = inputs.size()
            logpt = -self.loss_fn(inputs, target) #.long()
            pt = torch.exp(logpt)
            if alpha is not None:
                logpt *= alpha
            loss = -((1 - pt) ** gamma) * logpt
        return loss

    def FocalLoss_v2(self, inputs, target):
        if isinstance(inputs, (tuple, list)):
            tuple_len = len(inputs)
            # assert tuple_len == 2
            loss = 0
            for i in range(tuple_len):
                loss += self.loss_fn(inputs,target)
        else:
            loss = self.loss_fn(inputs, target)
        return loss

    def Binary(self, inputs, target):
        if isinstance(inputs, (tuple, list)):
            tuple_len = len(inputs)
            # assert tuple_len == 2
            loss = 0
            for i in range(tuple_len):
                if target.dim() == 3 and self.loss_type == 'bce':
                    target = self.convert_to_one_hot(target)
                loss_ = self.loss_fn(inputs[i], target)
                loss += loss_
        else:
            if target.dim() == 3 and self.loss_type == 'bce':
                target = self.convert_to_one_hot(target)
            loss = self.loss_fn(inputs, target)
        return loss

    def __call__(self, inputs, target, alpha=0.25, gamma=2):
        if self.loss_type == 'ce':
            return self.CrossEntropyLoss(inputs, target)
        elif self.loss_type == 'focal':
            return self.FocalLoss_v1(inputs, target)
        elif self.loss_type == 'focalv2':
            return self.FocalLoss_v2(inputs, target)
        elif self.loss_type == 'bce':
            return self.Binary(inputs, target)
        else:
            raise NotImplementedError

    def convert_to_one_hot(self, x):
        n, h, w = x.size()
        # remove the 255 index
        x[x == self.ignore_idx] = self.n_classes
        x = x.unsqueeze(1)

        # convert to one hot vector
        x_one_hot = torch.zeros(n, self.n_classes + 1, h, w).to(device=self.device)
        x_one_hot = x_one_hot.scatter_(1, x, 1)

        return x_one_hot[:, :self.n_classes, :, :].contiguous()


##
# version 1: use torch.autograd
class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
##
# version 2: user derived grad computation
# class FocalLossV2(nn.Module):
#
#     def __init__(self,
#                  alpha=0.25,
#                  gamma=2,
#                  reduction='mean'):
#         super(FocalLossV2, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, logits, label):
#         '''
#         Usage is same as nn.BCEWithLogits:
#             >>> criteria = FocalLossV2()
#             >>> logits = torch.randn(8, 19, 384, 384)
#             >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
#             >>> loss = criteria(logits, lbs)
#         '''
#         loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         if self.reduction == 'sum':
#             loss = loss.sum()
#         return loss
#
#
# class FocalSigmoidLossFuncV2(torch.autograd.Function):
#     '''
#     compute backward directly for better numeric stability
#     '''
#     @staticmethod
#     @amp.custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, logits, label, alpha, gamma):
#         #  logits = logits.float()
#
#         probs = torch.sigmoid(logits)
#         coeff = (label - probs).abs_().pow_(gamma).neg_()
#         log_probs = torch.where(logits >= 0,
#                 F.softplus(logits, -1, 50),
#                 logits - F.softplus(logits, 1, 50))
#         log_1_probs = torch.where(logits >= 0,
#                 -logits + F.softplus(logits, -1, 50),
#                 -F.softplus(logits, 1, 50))
#         ce_term1 = log_probs.mul_(label).mul_(alpha)
#         ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)
#         ce = ce_term1.add_(ce_term2)
#         loss = ce * coeff
#
#         ctx.vars = (coeff, probs, ce, label, gamma, alpha)
#
#         return loss
#
#     @staticmethod
#     @amp.custom_bwd
#     def backward(ctx, grad_output):
#         '''
#         compute gradient of focal loss
#         '''
#         (coeff, probs, ce, label, gamma, alpha) = ctx.vars
#
#         d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
#         d_coeff.mul_(probs).mul_(1. - probs)
#         d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
#         term1 = d_coeff.mul_(ce)
#
#         d_ce = label * alpha
#         d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
#         term2 = d_ce.mul(coeff)
#
#         grads = term1.add_(term2)
#         grads.mul_(grad_output)
#
#         return grads, None, None, None



if __name__ == '__main__':
    criteria = FocalLossV2()
    c2=FocalLossV1()
    logits = torch.randn(8, 19, 384, 384)
    lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
    loss = criteria(logits, lbs)
    loss2 = c2(logits, lbs)
    print(loss, loss2)
