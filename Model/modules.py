import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Utils.print_utils import print_info_message


def activation_fn(features, name='prelu'):
    '''
    :param features: # of features (only for PReLU)
    :param name: activation name (prelu, relu, selu)
    :param inplace: Inplace operation or not
    :return:
    '''
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'prelu':
        return nn.PReLU(features)
    elif name == 'lrelu':
        return nn.LeakyReLU(inplace=True)
    elif name == 'relu6':
        return nn.ReLU6(True)
    elif name == 'h_swish':
        return h_swish(True)
    elif name == 'gelu':
        return nn.GELU()
    else:
        NotImplementedError('Not implemented yet')
        exit()


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Attn(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, gamma=2, b=1, k_size=3):
        super(Attn, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class MSLRIBasic(nn.Module):
    def __init__(self,
                 inc: int,
                 ouc: int,
                 kernel_size: int = 1,
                 ratio: int = 2,
                 stride: int = 1,
                 act: bool = True,
                 act_name: str = 'gelu') -> None:
        super(MSLRIBasic, self).__init__()
        self.ouc = ouc
        init_channels = math.ceil(ouc / ratio)

        self.conv = nn.Sequential(
            nn.Conv2d(inc, init_channels, kernel_size, stride, kernel_size // 2, bias=True),
            nn.BatchNorm2d(init_channels),
            activation_fn(init_channels, act_name) if act else nn.Sequential()
        )

        self.LRMS = LRMS(init_channels, chunk=4)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.LRMS(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.ouc, :, :]


class MSLRI(nn.Module):
    def __init__(self,
                 inc: int,
                 ouc: int,
                 use_se: bool = True,
                 act_name: str = 'gelu') -> None:
        super(MSLRI, self).__init__()

        self.conv = nn.Sequential(
            MSLRIBasic(inc, ouc, kernel_size=1, act=True, act_name=act_name),
            Attn(ouc) if use_se else nn.Identity(),
            MSLRIBasic(ouc, ouc, kernel_size=1, act=True, act_name=act_name),
        )

    def forward(self, x):
        return self.conv(x) + x


class LDIBasic(nn.Module):
    def __init__(self,
                 inp: int,
                 oup: int,
                 kernel_size: int = 1,
                 ratio: int = 2,
                 dw_size: int = 3,
                 stride: int = 1,
                 act: bool = True,
                 act_name: str = 'gelu') -> None:
        super(LDIBasic, self).__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.compress = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            activation_fn(init_channels, act_name) if act else nn.Sequential()  # Relu
        )

        self.process = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),  #
            nn.BatchNorm2d(new_channels),
            activation_fn(new_channels, act_name) if act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.compress(x)
        x2 = self.process(x1)
        return torch.cat([x1, x2], dim=1)


class LDI(nn.Module):
    def __init__(self,
                 inc: int,
                 ouc: int,
                 use_se: bool = True,
                 act_name: str = 'gelu') -> None:
        super(LDI, self).__init__()

        self.conv = nn.Sequential(
            LDIBasic(inc, ouc, kernel_size=1, act=True, act_name=act_name),
            Attn(ouc) if use_se else nn.Sequential(),
            LDIBasic(ouc, ouc, kernel_size=1, act=True, act_name=act_name),  # False
        )

    def forward(self, x):
        return self.conv(x) + x


class LRMS(nn.Module):
    def __init__(self,
                 dim: int,
                 chunk: int = 4,
                 act_name: str = 'gelu') -> None:
        super(LRMS, self).__init__()
        self.chunk = chunk
        c = dim // chunk
        self.x1 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, dilation=1),
            nn.BatchNorm2d(c),
            activation_fn(c, name=act_name)
        )
        self.x2 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 2, groups=c, dilation=2),
            nn.BatchNorm2d(c),
            activation_fn(c, name=act_name)
        )
        self.x3 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 3, groups=c, dilation=3),
            nn.BatchNorm2d(c),
            activation_fn(c, name=act_name)
        )
        self.x4 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 4, groups=c, dilation=4),
            nn.BatchNorm2d(c),
            activation_fn(c, name=act_name)
        )

        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            activation_fn(dim, name=act_name)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
        )
        self.tail_act = activation_fn(dim, name=act_name)

    def forward(self, x):
        idn = x
        x = self.head(x)
        xs = torch.chunk(x, self.chunk, dim=1)
        ys = []
        for s in range(self.chunk):
            if s == 0:
                ys.append(self.x1(xs[s]))
            elif s == 1:
                ys.append(self.x2(xs[s] + ys[-1]))
            elif s == 2:
                ys.append(self.x3(xs[s] + ys[-1]))
            elif s == 3:
                ys.append(self.x4(xs[s] + ys[-1]))
        out = self.tail(torch.cat(ys, 1))
        return self.tail_act(out + idn)


class DWFF(nn.Module):
    def __init__(self,
                 in_channels: int,
                 height: int = 2,
                 reduction: int = 8,
                 bias: bool = False) -> None:
        super(DWFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2)
        )

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class MSSE(nn.Module):
    def __init__(self, inc, midc, ouc, act_name='gelu'):
        super(MSSE, self).__init__()
        self.x0 = nn.Sequential(
            nn.Conv2d(inc, midc, kernel_size=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.x1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(inc, midc, kernel_size=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.x2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.Conv2d(inc, midc, kernel_size=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.x3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.Conv2d(inc, midc, kernel_size=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.x4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inc, midc, kernel_size=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.p1 = nn.Sequential(
            nn.Conv2d(midc, midc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(midc, midc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(midc, midc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        self.p4 = nn.Sequential(
            nn.Conv2d(midc, midc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(midc),
            activation_fn(midc, act_name)
        )

        # self.head = nn.Sequential(
        #     nn.Conv2d(inc, inc, 1, 1, 0),
        #     nn.BatchNorm2d(inc),
        #     activation_fn(inc, name=act_name)
        # )

        self.tail = nn.Sequential(
            nn.BatchNorm2d(midc * 5),
            activation_fn(midc * 5, act_name),
            nn.Conv2d(midc * 5, ouc, kernel_size=1, bias=False),
            Attn(ouc)
        )
        # self.tail_act = activation_fn(ouc, name=act_name)

        self.identity = nn.Sequential(
            nn.BatchNorm2d(inc),
            activation_fn(inc, act_name),
            nn.Conv2d(inc, ouc, kernel_size=1, bias=False),
            Attn(ouc)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        idn = x
        # x = self.head(x)
        ys = []
        for s in range(5):
            if s == 0:
                ys.append(self.x0(x))
            elif s == 1:
                ys.append(
                    self.p1(F.interpolate(self.x1(x), size=(h, w), mode='bilinear') + ys[-1])
                )
            elif s == 2:
                ys.append(
                    self.p2(F.interpolate(self.x2(x), size=(h, w), mode='bilinear') + ys[-1])
                )
            elif s == 3:
                ys.append(
                    self.p3(F.interpolate(self.x3(x), size=(h, w), mode='bilinear') + ys[-1])
                )
            elif s == 4:
                ys.append(
                    self.p4(F.interpolate(self.x4(x), size=(h, w), mode='bilinear') + ys[-1])
                )

        out = self.tail(torch.cat(ys, dim=1)) + self.identity(idn)
        return out


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, act_name='gelu'):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.act = activation_fn(noutput, act_name)  # nn.ReLU6(inplace= True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut, act_name='gelu'):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = CBR(nIn + 3, nConv, 3, 2, 1, groups=1, act_name=act_name)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.bn_act = BR(nOut, act_name=act_name)
        self.input_project = InputInjection()

    def forward(self, input, image=None):
        bb = self.input_project(image, input)
        output = self.conv3x3(torch.cat([input, bb], dim=1))

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            avg_pool = self.avg_pool(input)
            output = torch.cat([output, max_pool + avg_pool], 1)

        output = self.bn_act(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input, target):
        h_input = input.size(2)
        h_tar = target.size(2)
        if h_tar == h_input:
            return input
        else:
            while True:
                input = F.avg_pool2d(input, kernel_size=3, padding=1, stride=2) + \
                        F.max_pool2d(input, kernel_size=3, padding=1, stride=2)
                h2 = input.size(2)
                if h2 == h_tar:
                    break
            return input


class Shuffle(nn.Module):
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1, act_name='prelu'):
        super().__init__()
        padding = int((kSize - 1) / 2) * dilation
        self.cbr = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(nOut),
            activation_fn(features=nOut, name=act_name)
        )

    def forward(self, x):
        return self.cbr(x)


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * dilation
        self.cb = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=1),
            nn.BatchNorm2d(nOut),
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.cb(x)


class BR(nn.Module):
    def __init__(self, nOut, act_name='prelu'):
        super().__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(nOut),
            activation_fn(nOut, name=act_name)
        )

    def forward(self, x):
        return self.br(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 data_format: str = "channels_first") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def mean_std(dataset='imagenet'):
    if dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset == 'cityscapes':
        mean = (0.284, 0.323, 0.282)
        std = (0.175, 0.180, 0.176)
    elif dataset == 'camvid':
        mean = (0.387, 0.400, 0.408)
        std = (0.267, 0.280, 0.278)
    return mean, std


def class_weight(dataset='cityscapes'):
    if dataset == 'cityscapes':
        print_info_message('Use Cityscapes class weights!')
        weight = torch.FloatTensor(
            [2.8149, 6.9850, 3.7890, 9.9428, 9.7702, 9.5110, 10.3113, 10.0264,
             4.6323, 9.5608, 7.8698, 9.5168, 10.3737, 6.6616, 10.2604, 10.2878,
             10.2898, 10.4053, 10.13809])
    elif dataset == 'camvid':
        print_info_message('Use Camvid class weights!')
        weight = torch.FloatTensor(
            [4.49689344, 3.30644627, 9.50793126, 2.9544884, 6.90970905, 5.08175348,
             9.33517863, 9.03942849, 6.64856547, 9.71374161, 9.5728822])

    return weight
