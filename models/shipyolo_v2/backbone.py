import torch
import torch.nn as nn

try:
    from models.shipyolo_v2.hx_function import Conv2dBatchMish, RCspRepBlock
except:
    from hx_function import Conv2dBatchMish, RCspRepBlock

class HX_RCSPDarknet(nn.Module):
    """
    example:
        import torch
        backbone = HX_RCSPDarknet(deploy=False)
        backbone = backbone.eval()
        inputs = torch.rand(1, 3, 416, 416)
        level_outputs = backbone(inputs)
        for level_out in level_outputs:
            print(tuple(level_out.shape))
        
        output:
            (1, 256, 52, 52)
            (1, 512, 26, 26)
            (1, 1024, 13, 13)
    """
    def __init__(self, deploy):
        super(HX_RCSPDarknet, self).__init__()
        # config
        self.inplanes = 32
        self.layers = (1, 2, 8, 8, 4)
        self.feature_channels = (64, 128, 256, 512, 1024)

        # conv0
        self.conv0 = Conv2dBatchMish(in_channels=3, out_channels=self.inplanes, kernel_size=3)

        # Layer
        self.layer_0 = RCspRepBlock(in_channels=self.inplanes, out_channels=self.feature_channels[0], res_nums=self.layers[0], first=True, deploy=deploy)
        self.layer_1 = RCspRepBlock(in_channels=self.feature_channels[0], out_channels=self.feature_channels[1], res_nums=self.layers[1], first=False, deploy=deploy)
        self.layer_2 = RCspRepBlock(in_channels=self.feature_channels[1], out_channels=self.feature_channels[2], res_nums=self.layers[2], first=False, deploy=deploy)
        self.layer_3 = RCspRepBlock(in_channels=self.feature_channels[2], out_channels=self.feature_channels[3], res_nums=self.layers[3], first=False, deploy=deploy)
        self.layer_4 = RCspRepBlock(in_channels=self.feature_channels[3], out_channels=self.feature_channels[4], res_nums=self.layers[4], first=False, deploy=deploy)

        # # Init weights, biases
        # hx_base.initialize_weights(self)

    def forward(self, x):
        x = self.conv0(x)

        x = self.layer_0(x)
        x = self.layer_1(x)

        out3 = self.layer_2(x)
        out4 = self.layer_3(out3)
        out5 = self.layer_4(out4)
        return out3, out4, out5  # 由大到小特征图输出


if __name__ == "__main__":
    import torch
    device = 'cuda:0'
    backbone = HX_RCSPDarknet(deploy=False)
    backbone = backbone.eval().to(device)
    inputs = torch.rand(1, 3, 416, 416).to(device)
    level_outputs = backbone(inputs)
    from tqdm import tqdm
    for i in tqdm(range(100)):
        level_outputs = backbone(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))