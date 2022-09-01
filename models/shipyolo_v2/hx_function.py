import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

# RFB
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=3, stride=stride, padding=1),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(3*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x1,x2),1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out*self.scale + x
        else:
            short = self.shortcut(x)
            out = out*self.scale + short
        out = self.relu(out)

        return out

class BasicRFA(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFA, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.conv_01 = BasicConv(in_planes, inter_planes, kernel_size=1, stride=1)
        self.branch0 = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=stride, padding=2),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=1, stride=1),
            BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=1, dilation=1, relu=False),
        )
        self.branch1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=9, stride=stride, padding=4),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False),
                )
        self.branch2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=13, stride=stride, padding=6),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, (inter_planes//2)*3, kernel_size=3, stride=stride, padding=1),
                BasicSepConv((inter_planes//2)*3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False),
                )

        self.ConvLinear = BasicConv((inter_planes//2)*3*3, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = self.relu = nn.LeakyReLU(inplace=False)
        # self.mish = Mish()

    def forward(self,x):
        x_conv_01 = self.conv_01(x)
        x0 = self.branch0(x_conv_01)
        x1 = self.branch1(x_conv_01)
        x2 = self.branch2(x_conv_01)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out*self.scale + x
        else:
            short = self.shortcut(x)
            out = out*self.scale + short
        out = self.relu(out)

        return out

class DSPP_V2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(DSPP_V2, self).__init__()

        # maxpool 
        self.maxpool_0 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        # dilation
        self.dila_conv_0 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dila_conv_1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dila_conv_2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=6, dilation=6)

        # 1 * 1conv
        self.conv = nn.Conv2d(in_planes * 4, in_planes, 1, 1)

    def forward(self, x):
        max0 = self.maxpool_0(x)
        max1 = self.maxpool_1(x)
        max2 = self.maxpool_2(x)

        dila0 = self.dila_conv_0(x)
        dila1 = self.dila_conv_1(x)
        dila2 = self.dila_conv_2(x)

        out_max = torch.cat((x, max0, max1, max2), 1)
        out_dila = torch.cat((x, dila0, dila1, dila2), 1)

        out = out_max + out_dila
        out = self.conv(out)
        return out


# RepVGGBlock
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def change_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy(),


class Mish(nn.Module):
    """
    activation funtion: mish
    """
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class Conv2dBatchMish(nn.Module):
    """
    conv2d + batchnorm2d + mish
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=1):
        super(Conv2dBatchMish, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if pad else 0

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            Mish()
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class Conv2dBatchLeaky(nn.Module):
    """
    conv2d + batchnorm2d + leaky
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if pad else 0

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class Add_UpFuseStage(nn.Module):
    """
    Add_UpFuseStage in AtFPN(ShipYOLO)
    input:
        x0, x1
    """
    def __init__(self, channels):
        super(Add_UpFuseStage, self).__init__()
        # Parameters
        self.channels = channels

        # conv
        self.conv0 = Conv2dBatchLeaky(in_channels=self.channels, out_channels=self.channels // 2, kernel_size=1, stride=1, pad=1)

        self.conv1 = nn.Sequential(
            Conv2dBatchLeaky(in_channels=self.channels, out_channels=self.channels // 2, kernel_size=1, stride=1, pad=1),
            nn.Upsample(scale_factor=2)
        ) 

    def forward(self, x):
        x0, x1 = x
        x0 = self.conv0(x0)
        return x0, torch.cat([x0, self.conv1(x1)], dim=1)

class UpFuseStage(nn.Module):
    """
    UpsampleFuse in PAFPN(YOLO-V4)
    input:
        x0, x1
    """
    def __init__(self, channels):
        super(UpFuseStage, self).__init__()
        # Parameters
        self.channels = channels

        # conv
        self.conv0 = Conv2dBatchLeaky(in_channels=self.channels, out_channels=self.channels // 2, kernel_size=1, stride=1, pad=1)

        self.conv1 = nn.Sequential(
            Conv2dBatchLeaky(in_channels=self.channels, out_channels=self.channels // 2, kernel_size=1, stride=1, pad=1),
            nn.Upsample(scale_factor=2)
        ) 

    def forward(self, x):
        x0, x1 = x
        return torch.cat([self.conv0(x0), self.conv1(x1)], dim=1)

class DownFuseStage(nn.Module):
    """
    DownsampleFuse in PAFPN(YOLO-V4)
    input:
        x0, x1
    """
    def __init__(self, channels):
        super(DownFuseStage, self).__init__()
        # Parameters
        self.channels = channels

        # downsample
        self.conv0 = Conv2dBatchLeaky(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=3, stride=2, pad=1)

    def forward(self, x):
        x0, x1 = x
        return torch.cat([self.conv0(x0), x1], dim=1)

class SppBlock(nn.Module):
    """
    Spp in yolov3/v4
    """
    def __init__(self):
        super(SppBlock, self).__init__()
        k = (5, 9, 13)
        self.maxpool0 = nn.MaxPool2d(kernel_size=k[0], stride=1, padding=(k[0] - 1) // 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=k[1], stride=1, padding=(k[1] - 1) // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=k[2], stride=1, padding=(k[2] - 1) // 2)

    def forward(self, x):
        return torch.cat([self.maxpool0(x), self.maxpool1(x), self.maxpool2(x), x], dim=1)

class ResBlock(nn.Module):
    """
    res Block in CspResBlock
    """
    def __init__(self, channels, hidden_channels=None):
        super(ResBlock, self).__init__()

        # Parameters
        self.channels = channels
        self.hidden_channels = hidden_channels
        if hidden_channels is None:
            self.hidden_channels = self.channels

        # Layer
        self.block = nn.Sequential(
            Conv2dBatchMish(in_channels=self.channels, out_channels=self.hidden_channels, kernel_size=1, stride=1),
            Conv2dBatchMish(in_channels=self.hidden_channels, out_channels=self.channels, kernel_size=3, stride=1)
        )

    def forward(self, x):
        return x + self.block(x)

class CspResBlock(nn.Module):
    """
    Block in CSPDarknet (YOLO-V4)
    input:
        in_channels: input channels 
        out_channels: output channels
        res_nums: cycle nums for res block in CSPResBlock
        first: first layer is not
    """
    def __init__(self, in_channels, out_channels, res_nums, first=None):
        super(CspResBlock, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_nums = res_nums

        # downsample conv
        self.conv0 = Conv2dBatchMish(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=2)
        
        if first:
            # split0 (res)
            self.split0_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
            self.split0_res = nn.Sequential(
                ResBlock(channels=self.out_channels, hidden_channels=self.out_channels // 2)
            )
            self.split0_conv1 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
            
            # split1
            self.split1_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)

            # concatenate conv
            self.conv1 = Conv2dBatchMish(in_channels=self.out_channels * 2, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            # split0 (res)
            self.split0_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels // 2, kernel_size=1, stride=1)
            self.split0_res = nn.Sequential(
                *[ResBlock(channels=self.out_channels // 2) for _ in range(self.res_nums)]
            )
            self.split0_conv1 = Conv2dBatchMish(in_channels=self.out_channels // 2, out_channels=self.out_channels // 2, kernel_size=1, stride=1)

            # split1
            self.split1_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels // 2, kernel_size=1, stride=1)

            # concatenate conv
            self.conv1 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)

        x0 = self.split0_conv0(x)
        x0 = self.split0_res(x0)
        x0 = self.split0_conv1(x0)

        x1 = self.split1_conv0(x)

        x = torch.cat([x0, x1], dim=1)
        x = self.conv1(x)
        
        return x

# cbam
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class RCspRepBlock(nn.Module):
    """
    Block in RCSPDarknet (ShipYOLO)
    input:
        in_channels: input channels 
        out_channels: output channels
        res_nums: cycle nums for res block in CSPResBlock
        deploy: False or True
    """
    def __init__(self, in_channels, out_channels, res_nums, first, deploy):
        super(RCspRepBlock, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_nums = res_nums

        # downsample conv
        self.conv0 = Conv2dBatchMish(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=2)

        if first:
            # split0 (rep)
            self.split0_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
            self.split0_rep = RepVGGBlock(in_channels=self.out_channels,
                                          out_channels=self.out_channels,
                                          kernel_size=3, stride=1, padding=1, deploy=deploy)
            self.split0_conv1 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)

            # split1
            self.split1_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)

            self.conv1 = Conv2dBatchMish(in_channels=self.out_channels * 2, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            # split0 (rep)
            self.split0_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels // 2, kernel_size=1, stride=1)
            self.split0_rep = nn.Sequential(
                *[RepVGGBlock(in_channels=self.out_channels // 2, 
                            out_channels=self.out_channels // 2, 
                            kernel_size=3, stride=1, padding=1, deploy=deploy) 
                            for _ in range(self.res_nums)]
            )
            self.split0_conv1 = Conv2dBatchMish(in_channels=self.out_channels // 2, out_channels=self.out_channels // 2, kernel_size=1, stride=1)

            # split1
            self.split1_conv0 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels // 2, kernel_size=1, stride=1)
        
            # concatenate conv
            self.conv1 = Conv2dBatchMish(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)

        x0 = self.split0_conv0(x)
        x0 = self.split0_rep(x0)
        x0 = self.split0_conv1(x0)

        x1 = self.split1_conv0(x)

        x = torch.cat([x0, x1], dim=1)
        x = self.conv1(x)
        
        return x