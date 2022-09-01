import torch
import torch.nn as nn

try:
    from models.shipyolo_v2.hx_function import Conv2dBatchLeaky, BasicRFA, Add_UpFuseStage, CBAM, DownFuseStage, DSPP_V2
except:
    from hx_function import Conv2dBatchLeaky, BasicRFA, Add_UpFuseStage, CBAM, DownFuseStage, DSPP_V2
    
class HX_AtFPN_V2(nn.Module):
    def __init__(self):
        super(HX_AtFPN_V2, self).__init__()

        self.layer0 = nn.Sequential(
            # CBL(3)
            Conv2dBatchLeaky(in_channels=1024, out_channels=512, kernel_size=1, stride=1, pad=1),
            Conv2dBatchLeaky(in_channels=512, out_channels=1024, kernel_size=3, stride=1, pad=1),
            Conv2dBatchLeaky(in_channels=1024, out_channels=512, kernel_size=1, stride=1, pad=1),
            # DSpp
            DSPP_V2(in_planes=512, out_planes=512),
            # CBL(3)
            Conv2dBatchLeaky(in_channels=512, out_channels=512, kernel_size=1, stride=1, pad=1),
            CBAM(512),
            Conv2dBatchLeaky(in_channels=512, out_channels=512, kernel_size=1, stride=1, pad=1)
        )

        self.layer1_a = Add_UpFuseStage(512)
        self.layer1_b = nn.Sequential(
            # Upsample 512 -> 256 then cat(256, 256) -> 512 
            # hx_base.UpFuseStage(512),
            # CBL(5)
            Conv2dBatchLeaky(in_channels=512, out_channels=256, kernel_size=1, stride=1, pad=1),
            CBAM(256),
            Conv2dBatchLeaky(in_channels=256, out_channels=256, kernel_size=1, stride=1, pad=1)
        )

        self.layer2_a = Add_UpFuseStage(256)
        self.layer2_b = nn.Sequential(
            # Upsample 256 -> 128 then cat(128, 128) -> 256 
            # hx_base.UpFuseStage(256),
            # CBAM
            Conv2dBatchLeaky(in_channels=256, out_channels=128, kernel_size=1, stride=1, pad=1),
            CBAM(128),
            Conv2dBatchLeaky(in_channels=128, out_channels=128, kernel_size=1, stride=1, pad=1)
        )
        self.layer3 = nn.Sequential(
            # Downsample 128 -> 256 then cat(256, 256) > 512
            DownFuseStage(128),
            # CBAM
            Conv2dBatchLeaky(in_channels=512, out_channels=256, kernel_size=1, stride=1, pad=1),
            CBAM(256),
            Conv2dBatchLeaky(in_channels=256, out_channels=256, kernel_size=1, stride=1, pad=1)
        )
        self.layer4 = nn.Sequential(
            # Downsample 256 -> 512 then cat(512, 512) > 1024
            DownFuseStage(256),
            # CBAM
            Conv2dBatchLeaky(in_channels=1024, out_channels=512, kernel_size=1, stride=1, pad=1),
            CBAM(512),
            Conv2dBatchLeaky(in_channels=512, out_channels=512, kernel_size=1, stride=1, pad=1)
        )

        # hx_base.initialize_weights(self)

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

        return [out3, out4, out5]

if __name__ == "__main__":
    import torch
    device = 'cuda:0'
    inputs1 = torch.rand(1, 256, 52, 52).to(device)
    inputs2 = torch.rand(1, 512, 26, 26).to(device)
    inputs3 = torch.rand(1, 1024, 13, 13).to(device)
    inputs_list = [inputs1, inputs2, inputs3]
    neck = HX_AtFPN().to(device)
    output = neck(inputs_list)

    