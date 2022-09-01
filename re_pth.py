import torch

from tqdm import tqdm
import copy
import numpy as np

# from models.shipyolo.shipyolo import ShipYOLO
from models.shipyolo_v2.shipyolo_v2 import ShipYOLO_V2

device = 'cuda:0'

# 参考代码：https://github.com/bofen97

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'change_to_deploy'):
            module.change_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

if __name__ == "__main__":

    # 原模型读取
    model = ShipYOLO_V2(num_classes=14, deploy=False, torch_trt=False).eval()
    model.load_state_dict(torch.load("runs/shipyolo_v2/best.pt", map_location=device)['model'])
    model = model.to(device)

    deploy_model = repvgg_model_convert(model, "runs/shipyolo_v2/best_deploy.pth")

    # # 参数化重构
    model = ShipYOLO_V2(num_classes=14, deploy=True, torch_trt=False).eval()
    model.load_state_dict(torch.load("runs/shipyolo_v2/best_deploy.pth", map_location=device))
    model = model.to(device)

