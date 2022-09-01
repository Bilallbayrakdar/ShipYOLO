import torch
import torch.nn as nn
# from torch.nn.functional import pad

try:
    from models.shipyolo_v2.backbone import HX_RCSPDarknet
    from models.shipyolo_v2.neck import HX_AtFPN_V2
    from models.shipyolo_v2.head import YOLOLayer
    from models.shipyolo_v2.hx_function import Conv2dBatchLeaky, initialize_weights
except:
    from backbone import HX_RCSPDarknet
    from neck import HX_AtFPN_V2
    from head import YOLOLayer
    from hx_function import Conv2dBatchLeaky, initialize_weights

class ShipYOLO_V2(nn.Module):
    def __init__(self, num_classes, deploy, torch_trt=False):
        super(ShipYOLO_V2, self).__init__()

        # 参数
        #------yololayer------# voc anchor
        anchors = [
            [
                [12, 16],
                [19, 36],
                [40, 28]
            ],
            [
                [36, 75],
                [76, 55],
                [72, 146]
            ],
            [
                [142, 110],
                [192, 243],
                [459, 401]
            ]
        ]
        # kmean anchor huawei
        # anchors = [
        #     [
        #         [67, 113],
        #         [201, 116],
        #         [18, 29]
        #     ],
        #     [
        #         [8, 14],
        #         [426, 317],
        #         [38, 54]
        #     ],
        #     [
        #         [415, 184],
        #         [207, 366],
        #         [107, 232]
        #     ]
        # ]
        yolo_index = [0, 1, 2]
        stride = [8, 16, 32]
        self.torch_trt = torch_trt
        #------yololayer------#

        head_channel = [256, 512, 1024]
        out_channel = (num_classes + 5) * 3

        self.backbone = HX_RCSPDarknet(deploy=deploy)
        self.neck = HX_AtFPN_V2()

        # head0 (out3)
        self.conv_head0 = nn.Sequential(
            Conv2dBatchLeaky(in_channels=128, out_channels=head_channel[0], kernel_size=3, stride=1, pad=1),
            nn.Conv2d(in_channels=head_channel[0], out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        )
        self.head0 = YOLOLayer(anchors=anchors[0],  # anchor list
                                nc=num_classes,  # number of classes
                                yolo_index=yolo_index[0],  # 0, 1, 2...
                                stride=stride[yolo_index[0]])

        # head1 (out4)
        self.conv_head1 = nn.Sequential(
            Conv2dBatchLeaky(in_channels=256, out_channels=head_channel[1], kernel_size=3, stride=1, pad=1),
            nn.Conv2d(in_channels=head_channel[1], out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        )
        self.head1 = YOLOLayer(anchors=anchors[1],  # anchor list
                                nc=num_classes,  # number of classes
                                yolo_index=yolo_index[1],  # 0, 1, 2...
                                stride=stride[yolo_index[1]])

        # head2 (out5)
        self.conv_head2 = nn.Sequential(
            Conv2dBatchLeaky(in_channels=512, out_channels=head_channel[2], kernel_size=3, stride=1, pad=1),
            nn.Conv2d(in_channels=head_channel[2], out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        )
        self.head2 = YOLOLayer(anchors=anchors[2],  # anchor list
                                nc=num_classes,  # number of classes
                                yolo_index=yolo_index[2],  # 0, 1, 2...
                                stride=stride[yolo_index[2]])
        # Init weights, biases
        initialize_weights(self)
    
    def forward(self, x):
        out3, out4, out5 = self.backbone(x)
        out3, out4, out5 = self.neck((out3, out4, out5))

        out3 = self.conv_head0(out3)
        out4 = self.conv_head1(out4)
        out5 = self.conv_head2(out5)

        
        # return [out3, out4, out5]
        
        yolo_out = [self.head0(out3), self.head1(out4), self.head2(out5)]
        # return yolo_out 
        if self.training:
            return yolo_out
        else:
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p

if __name__ == "__main__":
    import torch
    device = 'cuda:0'
    model = ShipYOLO(num_classes=2, deploy=False)
    model = model.eval().to(device)
    # inputs = torch.rand(1, 3, 416, 416).to(device)
    # output = model(inputs)
    # print(output.shape)
    anchor_vec_list = [model.head0.anchor_vec, model.head1.anchor_vec, model.head2.anchor_vec]
    print(anchor_vec_list[0])
    print(anchor_vec_list[1])
    print(anchor_vec_list[2])