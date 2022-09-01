import torch
import torch.nn as nn

from ..utils import hx_function as hx_base

from mmcv.runner import BaseModule
from ..builder import NECKS

@NECKS.register_module()
class HX_AtFPN(BaseModule):
    def __init__(self):
        super(HX_AtFPN, self).__init__()

        self.layer0 = nn.Sequential(
            # CBL(3)
            hx_base.Conv2dBatchLeaky(in_channels=1024, out_channels=512, kernel_size=1, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=512, out_channels=1024, kernel_size=3, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=1024, out_channels=512, kernel_size=1, stride=1, pad=1),
            # Spp
            hx_base.BasicRFA(in_planes=512, out_planes=512),
            # CBL(3)
            hx_base.Conv2dBatchLeaky(in_channels=512, out_channels=512, kernel_size=1, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=512, out_channels=1024, kernel_size=3, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=1024, out_channels=512, kernel_size=1, stride=1, pad=1),
        )

        self.layer1_a = hx_base.Add_UpFuseStage(512)
        self.layer1_b = nn.Sequential(
            # Upsample 512 -> 256 then cat(256, 256) -> 512 
            # hx_base.UpFuseStage(512),
            # CBL(5)
            hx_base.Conv2dBatchLeaky(in_channels=512, out_channels=256, kernel_size=1, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=256, out_channels=512, kernel_size=3, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=512, out_channels=256, kernel_size=1, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=256, out_channels=512, kernel_size=3, stride=1, pad=1),
            hx_base.Conv2dBatchLeaky(in_channels=512, out_channels=256, kernel_size=1, stride=1, pad=1),
        )

        self.layer2_a = hx_base.Add_UpFuseStage(256)
        self.layer2_b = nn.Sequential(
            # Upsample 256 -> 128 then cat(128, 128) -> 256 
            # hx_base.UpFuseStage(256),
            # CBAM
            hx_base.Conv2dBatchLeaky(in_channels=256, out_channels=128, kernel_size=1, stride=1, pad=1),
            hx_base.CBAM(128)
        )
        self.layer3 = nn.Sequential(
            # Downsample 128 -> 256 then cat(256, 256) > 512 DownFuseStage
            hx_base.DownFuseStage(128),
            # CBAM
            hx_base.Conv2dBatchLeaky(in_channels=512, out_channels=256, kernel_size=1, stride=1, pad=1),
            hx_base.CBAM(256)
        )
        self.layer4 = nn.Sequential(
            # Downsample 256 -> 512 then cat(512, 512) > 1024 DownFuseStage
            hx_base.DownFuseStage(256),
            # CBAM
            hx_base.Conv2dBatchLeaky(in_channels=1024, out_channels=512, kernel_size=1, stride=1, pad=1),
            hx_base.CBAM(512)
        )

        hx_base.initialize_weights(self)

    def forward(self, x):
        out3, out4, out5 = x
        
        out5 = self.layer0(out5)
        add5 = out5

        add4, out4 = self.layer1_a([out4, out5])
        out4 = self.layer1_b(out4)

        add3, out3 = self.layer2_a([out3, out4])
        out3 = self.layer2_b(out3) + add3

        out4 = self.layer3([out3, out4]) + add4
        out5 = self.layer4([out4, out5]) + add5

        return tuple([out5, out4, out3])