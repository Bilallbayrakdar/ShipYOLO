import torch
import torch.nn as nn

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, yolo_index, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.stride = stride  # layer stride
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            io = p.sigmoid()
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            #io = p.clone()  # inference output
            #io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            #io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            #io[..., :4] *= self.stride
            #torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

if __name__ == "__main__":
    import torch
    device = 'cuda:0'
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
    num_classes = 2
    # img_size = (416, 416)
    yolo_index = [0, 1, 2]
    stride = [8, 16, 32]

    if len(anchors) != len(stride):
        print("anchors 和 stride 不一致")
        exit()
    
    for i in range(len(anchors)):
        modules = YOLOLayer(anchors=anchors[i],  # anchor list
                                nc=num_classes,  # number of classes
                                yolo_index=yolo_index[i],  # 0, 1, 2...
                                stride=stride[yolo_index[i]]).to(device)
        inputs = torch.rand(1, 21, 13, 13).to(device)
        modules = modules.train()
        y = modules(inputs)
        print(y.shape)
        exit()