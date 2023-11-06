import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.modules import *


class PMSDSEN(nn.Module):
    def __init__(self,
                 cls: int = 19,
                 act_name: str = 'gelu') -> None:
        super(PMSDSEN, self).__init__()
        blocks = [1, 2, 6, 2, 1]
        init_out_c = 32

        self.DWFF4 = DWFF(init_out_c * 2, reduction=8, bias=True)
        self.DWFF5 = DWFF(init_out_c * 1, reduction=4, bias=True)

        ''' Init Conv: h,w,c '''
        self.Stem_Conv = nn.Sequential(
            nn.Conv2d(3, init_out_c // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(init_out_c // 2),
            activation_fn(init_out_c // 2, act_name),
            nn.Conv2d(init_out_c // 2, init_out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(init_out_c)
        )

        ''' ----------------------------------- Encoder ----------------------------------- '''

        ''' Block1: down 1 + basic blocks[1]-1   Size: h/2,w/2,c -> h/2,w/2,2c '''
        self.layer1 = nn.Sequential(*[PMSDSE(init_out_c, init_out_c, act_name=act_name) for _ in range(blocks[0])])

        ''' Block2: down 1 + basic blocks[2]-1   Size: h/2,w/2,2c -> h/4,w/4,4c '''
        self.down2 = DownSamplingBlock(init_out_c, init_out_c * 2, act_name=act_name)
        self.layer2 = nn.Sequential(
            *[PMSDSE(init_out_c * 2, init_out_c * 2, act_name=act_name) for _ in range(blocks[1])])

        ''' Block3: down 1 + basic blocks[3]-1   Size: h/4,w/4,4c -> h/8,w/8,8c '''
        self.down3 = DownSamplingBlock(init_out_c * 2, init_out_c * 4, act_name=act_name)
        self.layer3 = nn.Sequential(
            *[PMSDSE(init_out_c * 4, init_out_c * 4, act_name=act_name) for _ in range(blocks[2])])

        ''' ----------------------------------- Decoder ----------------------------------- '''
        ''' Block6: basic blocks[6]   Size: h/8,w/8,8c -> h/4,w/4,4c '''
        self.up4 = UpsamplerBlock(init_out_c * 4, init_out_c * 2, act_name=act_name)
        self.layer4 = nn.Sequential(
            *[PMSDSE(init_out_c * 2, init_out_c * 2, act_name=act_name) for _ in range(blocks[3])])

        ''' Block7: basic blocks[7]   Size: h/4,w/4,4c -> h/2,w/2,2c '''
        self.up5 = UpsamplerBlock(init_out_c * 2, init_out_c * 1, act_name=act_name)
        self.layer5 = nn.Sequential(*[PMSDSE(init_out_c, init_out_c, act_name=act_name) for _ in range(blocks[4])])

        self.MSSE_ = MSSE(init_out_c * 4, init_out_c * 2, init_out_c * 4, act_name=act_name)

        self.seg_head_x2 = nn.Sequential(
            nn.Conv2d(init_out_c, init_out_c // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_out_c // 2),
            activation_fn(init_out_c // 2, act_name),
            nn.Conv2d(init_out_c // 2, cls, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(cls)
        )
        self.seg_head_x4 = nn.Sequential(
            nn.Conv2d(init_out_c * 2, init_out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_out_c),
            activation_fn(init_out_c, act_name),
            nn.Conv2d(init_out_c, cls, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(cls)
        )
        self.seg_head_x8 = nn.Sequential(
            nn.Conv2d(init_out_c * 4, init_out_c * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(init_out_c * 2),
            activation_fn(init_out_c * 2, act_name),
            nn.Conv2d(init_out_c * 2, cls, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(cls)
        )

    def forward(self, input):
        ''' hw3 -> hwc '''
        stem = self.Stem_Conv(input)  # [3,512,1024] -> [64,256,512]

        layer1 = self.layer1(stem)  # [64,256,512]

        down2 = self.down2(layer1, input)  # [64,256,512] -> [128,128,256]
        layer2 = self.layer2(down2)  # [128,128,256]

        down3 = self.down3(layer2, input)  # [128,128,256] -> [256,64,128]
        layer3 = self.layer3(down3)

        layer3 = self.MSSE_(layer3)  # [256,64,128]

        up4 = self.up4(layer3)  # [256,64,128] -> [128,128,256]
        layer4 = self.layer4(self.DWFF4([up4, layer2]))  # [128,128,256]

        up5 = self.up5(layer4)  # [128,128,256] -> [64,256,512]
        layer5 = self.layer5(self.DWFF5([up5, layer1]))  # [64,256,512]

        return F.interpolate(self.seg_head_x2(layer5), scale_factor=2, mode='bilinear'), \
               F.interpolate(self.seg_head_x4(layer4), scale_factor=4, mode='bilinear'), \
               F.interpolate(self.seg_head_x8(layer3), scale_factor=8, mode='bilinear')

    def get_basenet_params(self):
        modules_base = [self.Stem_Conv, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
                        self.down2, self.down3, self.up4, self.up5, self.MSSE, self.DWFF4, self.DWFF5]
        for i in range(len(modules_base)):
            for m in modules_base[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_segment_params(self):
        modules_seg = [self.seg_head_x2, self.seg_head_x4, self.seg_head_x8]
        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class PMSDSE(nn.Module):
    def __init__(self,
                 inp: int,
                 oup: int,
                 act_name: str = 'gelu') -> None:
        super(PMSDSE, self).__init__()
        self.oup = oup

        self.LDI = LDI(inp, inp, use_se=True, act_name=act_name)
        self.MSLRI = MSLRI(inp, inp, use_se=True, act_name=act_name)

        self.shuffle = Shuffle(groups=inp)
        self.Attn = Attn(inp)
        self.bn_act = BR(inp, act_name=act_name)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inp, inp, 1, 1, 0),
            nn.BatchNorm2d(inp)
        )
        self.Attn_output = Attn(inp)
        self.entry_bn_act = BR(inp, act_name=act_name)
        self.act_out = activation_fn(inp, act_name)

    def forward(self, input):
        fea = input
        input = self.entry_bn_act(input)

        output = self.LDI(input) + self.MSLRI(input) + self.Attn(input)
        output = self.Attn_output(self.conv1x1(self.bn_act(output)))
        output = self.shuffle(self.act_out(output + fea))

        return output


if __name__ == '__main__':
    from Metrics.flops_compute import compute_flops, compute_parameters

    h, w = 512, 1024
    x = torch.randn(2, 3, h, w).cuda()

    with torch.no_grad():
        model = PMSDSEN(19, act_name='gelu').cuda()
        # print(model)
        out, x4, x8 = model(x)
        print(x4.shape, x8.shape, out.shape)

        flops = compute_flops(model, x)
        params = compute_parameters(model)
        print(f'FLOPS: {flops} G | Params: {params} M')

'''
torch.Size([2, 19, 512, 1024]) torch.Size([2, 19, 512, 1024]) torch.Size([2, 19, 512, 1024])
FLOPS: 10.2669418 G | Params: 0.923003 M

# New MSSE
torch.Size([2, 19, 512, 1024]) torch.Size([2, 19, 512, 1024]) torch.Size([2, 19, 512, 1024])
FLOPS: 10.267466088 G | Params: 0.922107 M
FLOPS: 10.401683816 G | Params: 0.939141 M

'''
