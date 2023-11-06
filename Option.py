import argparse
import time
from Utils.print_utils import print_error_message, print_log_message, print_info_message

parser = argparse.ArgumentParser()
''' 1. Gloab Settings '''
parser.add_argument('--wandb', action='store_true', default=False, help='Whether to use Wandb')
parser.add_argument('--wandb_project', type=str, default='Default', help='Wandb project name')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Whether to use cuda')
parser.add_argument('--gpu_ids', type=str, default='0',
                    help='use which gpu to train, must be a comma-separated list of integers only (default=0)')

''' 2. Saver '''
parser.add_argument('--exp_name', type=str, default=None, help='save directory name')

''' 3. Dataset '''
parser.add_argument('--train_batchsize', type=int, default=4, help='train batchsize')
parser.add_argument('--val_batchsize', type=int, default=4, help='Val batchsize')
parser.add_argument('--crop_size', type=int, nargs='+', default=[1024, 512],
                    help='Image Crop size (w,h) list of image crop sizes, with each item storing the crop size (should be a tuple).')
parser.add_argument('--rootpath', type=str, default=r'',
                    help='Datset rootpath')
# parser.add_argument('--rescale', type=int, default=4, help='down/up scale')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'cityscapes', 'camvid'],
                    help='Datasets')

''' 4. Loss '''
parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                    help='whether to use balanced balance?')
parser.add_argument('--loss_type', default='ce', choices=['ce', 'focal', 'ohem'], help='Loss function (ce or miou)')

''' 5. Learning rate '''
parser.add_argument('--lr', type=float, default=0.045, help='initial learning rate')
parser.add_argument('--lr_scheduler', default='hybrid',
                    choices=['cos', 'poly', 'step', 'fixed', 'clr', 'linear', 'hybrid'],
                    help='Learning rate scheduler (fixed, clr, poly)')
parser.add_argument('--warmup_epochs', type=int, default=0, help='Warm Up epoch')
# CyclicLR
parser.add_argument('--step_sizes', type=int, default=51, help='steps at which lr should be decreased')
parser.add_argument('--lr_decay', type=float, default=0.5, help='factor by which lr should be decreased')
# HybirdLR
parser.add_argument('--clr_max', type=int, default=61,
                    help='Max number of epochs for cylic LR before changing last cycle to linear')
parser.add_argument('--cycle_len', type=int, default=5, help='Duration of cycle')

''' 6. Optimizer '''
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer choose')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='SGD weight_decay')  # 5e-4

''' 7. Train Settings '''
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--epochs', type=int, default=500, help='Total epochs')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Finetune the segmentation model')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='Finetune the segmentation model')

''' 8. Model '''
parser.add_argument('--model', default='',
                    help='Which model? basic= basic CNN model, res=resnet style)')

args = parser.parse_args()

if args.dataset == 'pascal':
    args.scale = (0.5, 2.0)
elif args.dataset == 'cityscapes' or 'camvid':
    if args.crop_size[0] == 512:
        args.scale = (0.25, 0.5)
    elif args.crop_size[0] == 1024:
        args.scale = (0.35, 1.0)
    elif args.crop_size[0] == 2048:
        args.scale = (0.75, 2.0)  # (1.0, 2.0)
    elif args.crop_size[0] == 256:
        args.scale = (0.125, 0.25)  # (0.75, 2.0)#(0.25, 0.5)
    elif args.crop_size[0] == 480:  # for camvid
        args.scale = (0.35, 1.0)
    elif args.crop_size[0] == 960:  # for camvid
        args.scale = (0.5, 2.5)  # (0.75, 2.0)
    else:
        print_error_message('Select image size from 512x256, 1024x512, 2048x1024')
    print_log_message('Using scale = ({}, {})'.format(args.scale[0], args.scale[1]))
else:
    print_error_message('{} dataset not yet supported'.format(args.dataset))

assert len(args.crop_size) == 2, 'crop-size argument must contain 2 values'
args.crop_size = tuple(args.crop_size)
args.directory = f'{args.model}_{args.dataset}'
args.exp_name = f'{args.crop_size[0]}_{args.crop_size[1]}_{args.scale[0]}_{args.scale[1]}_{args.lr}_{args.lr_scheduler}_{args.loss_type}_{args.exp_name}'  # .format(args.savedir, args.model, args.dataset,
args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]

