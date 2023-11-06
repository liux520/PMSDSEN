import numpy as np
from tqdm import tqdm
import wandb
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision

# Utils
from Utils.print_utils import print_info_message
from Utils.msic import AverageMeter, set_seed
from Utils.Lr_scheduler import LR_Scheduler_Head
from Utils.Saver import MySaver
from Metrics.metrics import Evaluator
from Metrics.flops_compute import compute_flops, compute_parameters

# Loss
from Loss.Loss import SegmentationLoss
from Utils.Parallel import DataParallelModel, DataParallelCriterion
from Model.modules import class_weight, LayerNorm

warnings.filterwarnings("ignore")
os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "dryrun"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_info_message(f'Using {self.device}!')
        ''' ------------------------------------- Wandb ------------------------------------- '''
        if self.args.wandb:
            print_info_message(f'Using Wandb!')
            wandb.init(project=self.args.wangb_project, entity="")

        ''' ------------------------------------- Saver ------------------------------------- '''
        self.saver = MySaver(directory=self.args.directory, exp_name=self.args.exp_name)

        ''' ----------------------------------- Dataloader ---------------------------------- '''
        if self.args.dataset == 'cityscapes':
            from Data.cityscapes import CityDataloader
            kwargs = {'num_workers': self.args.workers, 'pin_memory': False}
            self.train_loader, self.val_loader, self.test_loader, self.nclass, self.ignore_label = \
                CityDataloader(self.args.train_batchsize, self.args.val_batchsize, self.args.crop_size,
                               sr_scale=2, rescale=self.args.scale, **kwargs)
            print_info_message(f'Training on {self.args.dataset} dataset, Train_loader_len:{len(self.train_loader)}, '
                               f'Val_loader_len:{len(self.val_loader)}, Classes:{self.nclass}')

        elif self.args.dataset == 'camvid':
            from Data.camvid import CamvidDataloader
            kwargs = {'num_workers': self.args.workers, 'pin_memory': True}
            self.train_loader, self.val_loader, self.test_loader, self.nclass, self.ignore_label = \
                CamvidDataloader(self.args.train_batchsize, self.args.val_batchsize, self.args.crop_size,
                                 sr_scale=2, rescale=self.args.scale, **kwargs)
            print_info_message(f'Training on {self.args.dataset} dataset, Train_loader_len:{len(self.train_loader)}, '
                               f'Val_loader_len:{len(self.val_loader)}, Classes:{self.nclass}')

        ''' ------------------------------------ Network ------------------------------------ '''
        from Model.model_import import model_import
        self.model = model_import('PMSDSEN', self.nclass, 'gelu', False)
        if self.args.wandb: wandb.watch(self.model, log='all')
        with torch.no_grad():
            parameters = compute_parameters(self.model)
            flops = compute_flops(self.model, input=torch.Tensor(1, 3, 512, 1024))
        print_info_message(f'Model Parameters:{parameters:,}, Flops:{flops:,} of input size:{512}x{1024}')
        self.saver.save_configs(self.args, parameters, flops)

        ''' ----------------------------------- Optimizer ----------------------------------- '''
        if self.args.optimizer == 'SGD':
            train_params = [{'params': self.model.get_basenet_params(), 'lr': self.args.lr},
                            {'params': self.model.get_segment_params(), 'lr': self.args.lr * 10}]  # get_10x_lr_params
            self.optimizer = torch.optim.SGD(train_params, momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay, nesterov=False,
                                             lr=self.args.lr)  # 1e-4  4e-5  5e-4
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              self.args.lr, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)

        ''' ----------------------------------- Criterion ----------------------------------- '''
        from Loss.Loss import OhemCrossEntropy
        # whether to use class balanced weights
        if self.args.use_balanced_weights:
            print_info_message('Using balanced weights')
            weight = class_weight(dataset=self.args.dataset)
        else:
            weight = None
        if self.args.loss_type == 'ce':
            self.criterion = SegmentationLoss(n_classes=self.nclass, loss_type='ce', device=self.device,
                                              ignore_idx=self.ignore_label, class_wts=weight)
        elif self.args.loss_type == 'focal':
            self.criterion = None  # FocalLossV2(alpha=0.25, gamma=2, )
        elif self.args.loss_type == 'ohem':
            self.criterion = OhemCrossEntropy(self.ignore_label, 0.9, weight,
                                              self.args.train_batchsize, self.args.crop_size)
        ''' ----------------------------------- Evaluator ----------------------------------- '''
        self.evaluator = Evaluator(self.nclass)

        ''' ----------------------------------- Scheduler ----------------------------------- '''
        # ['cos','poly','step']
        self.scheduler = LR_Scheduler_Head(self.args.lr_scheduler, self.args.lr, self.args.epochs,
                                           len(self.train_loader), warmup_epochs=self.args.warmup_epochs)
        # ['CyclicLR', 'LinearLR', 'HybirdLR']
        # step_size = self.args.step_sizes
        # step_sizes = [step_size * i for i in range(1, int(math.ceil(args.epochs / step_size)))]
        # self.scheduler2 = HybirdLR(base_lr=self.args.lr,clr_max=self.args.clr_max, max_epochs=self.args.epochs, cycle_len=self.args.cycle_len)
        # self.scheduler3 = CyclicLR(min_lr=self.args.lr,cycle_len=5,steps=step_sizes,gamma=self.args.lr_decay,step=True)#0.5

        ''' ------------------------------------- Cuda -------------------------------------- '''
        if self.args.use_cuda:
            if len(self.args.gpu_ids) == 1:
                print_info_message(f'Using Single GPU')
                # for a single GPU, we do not need DataParallel wrapper for Criteria. So, falling back to its internal wrapper
                self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids).to(self.device)
                self.criterion = self.criterion.to(self.device)
            elif len(self.args.gpu_ids) > 1:
                print_info_message(f'Using {self.args.gpu_ids} GPU')
                self.model = DataParallelModel(self.model, device_ids=self.args.gpu_ids).to(self.device)
                self.criterion = DataParallelCriterion(self.criterion, device_ids=self.args.gpu_ids).to(self.device)

        ''' ------------------------------ Resuming checkpoint ------------------------------ '''
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            ###
            model_dict = self.model.module.state_dict()
            overlap_ = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(overlap_)
            print_info_message(f'{(len(overlap_) * 1.0 / len(checkpoint["state_dict"]) * 100):.4f}% is loaded!')
            print([k for k, v in checkpoint['state_dict'].items() if k in model_dict])
            ###
            if self.args.use_cuda:
                self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(model_dict)

            if not self.args.finetune:
                self.args.start_epoch = checkpoint['epoch'] + 1
                self.best_pred = checkpoint['best_pred']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print_info_message('Resume mode!')
            elif self.args.finetune:
                self.args.start_epoch = 0
                self.best_pred = 0.0
                print_info_message(f'Finetune mode!')
            print_info_message(
                f"loaded checkpoint:'{self.args.resume}' epoch:{checkpoint['epoch']} previous_best_pred:{checkpoint['best_pred']}")
            print_info_message([k for k, v in checkpoint.items()])

        ''' ----------------------------------- Freeze BN ----------------------------------- '''
        if self.args.freeze_bn:
            print_info_message('Freezing batch normalization layers!')
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, LayerNorm)):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def training(self, epoch):
        train_losses = AverageMeter()
        self.model.train()
        tbar = tqdm(self.train_loader)
        for i, (sample, _) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            image, target = sample['image'], sample['label']
            if epoch == 0 and i == 0:
                print_info_message(f'image size:{image.shape} label_size:{target.shape}')

            if self.args.use_cuda:
                image, target = image.to(self.device), target.to(self.device)
            out, out_aux1, out_aux2 = self.model(image)

            loss = self.criterion(out, target) + 0.1 * self.criterion(out_aux1, target) + 0.1 * self.criterion(out_aux2,
                                                                                                               target)
            train_losses.update(loss.item(), n=image.shape[0])
            if self.args.wandb: wandb.log({"train_loss_iter": loss.item()})
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tbar.set_description(f'Train loss:{train_losses.avg:.4f} Loss_ss:{loss.item():.4f}')

        print_info_message('Epoch:{}, numImages:{}, Train_loss:{:.3f}'.format(epoch, i * self.args.train_batchsize +
                                                                              image.data.shape[0], train_losses.avg))
        if self.args.wandb: wandb.log({"train_total_loss": train_losses.avg})
        self.saver.save_record_train(epoch, train_losses.avg)

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        lr = self.optimizer.param_groups[0]['lr']
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_losses = AverageMeter()
        for i, (sample, _) in enumerate(tbar):
            image, target = sample['image'], sample['label']
            # image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=True)
            if epoch == 0 and i == 0:
                print_info_message(f'image size:{image.shape} label_size:{target.shape}')
            # self.saver.save_images_input_label(image,target,i,1000,epoch, denormal=True)

            if self.args.use_cuda:
                image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                output, out_aux1, out_aux2 = self.model(image)
                # output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
                # self.saver.save_ss(i, 1000, output, epoch)

            loss = self.criterion(output, target)
            test_losses.update(loss.item(), n=image.shape[0])
            tbar.set_description('Test loss: {:.3f}'.format(test_losses.avg))

            pred = output[0].data.cpu().numpy() if isinstance(output, (list, tuple)) else output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print_info_message(
            'Epoch:{}, numImages:{}, Loss:{:.3f}'.format(epoch, i * self.args.train_batchsize + image.data.shape[0],
                                                         test_losses.avg))
        print_info_message(
            "Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwIoU: {:.4f}".format(Acc, Acc_class, mIoU, FWIoU))
        if self.args.wandb:
            wandb.log({
                "val_total_loss": test_losses.avg,
                "mIoU": mIoU,
                "fwIoU": FWIoU,
                "Acc": Acc,
                "Acc_class": Acc_class})

        self.saver.save_record_val(epoch, lr, test_losses.avg, mIoU, 0, 0, FWIoU, Acc, Acc_class)

        new_pred = mIoU
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'pred': mIoU,
        }
        self.saver.save_checkpoint_override(state, epoch, new_pred, self.best_pred, self.args)
        if new_pred > self.best_pred:
            self.best_pred = new_pred


if __name__ == '__main__':
    from Option import args

    set_seed(123)

    trainer = Trainer(args)
    print('start epoch:{} total epoch:{}'.format(trainer.args.start_epoch, trainer.args.epochs))
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
