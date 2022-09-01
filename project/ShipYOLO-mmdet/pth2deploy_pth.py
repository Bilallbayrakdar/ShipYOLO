import torch
import mmcv
import mmdet
from tqdm import tqdm
import copy

from mmdet.apis import init_detector, inference_detector
from mmdet.models import build_detector

import numpy as np

device = 'cuda:0'

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
    config_file = 'work_dirs_512/shipyolo_v2/shipyolo_v2.py'
    checkpoint_file = 'work_dirs_512/shipyolo_v2/epoch_260.pth'
    config_deploy_file = 'work_dirs_512/shipyolo_v2/shipyolo_v2_deploy.py'
    save_checkpoint_file = 'work_dirs_512/shipyolo_v2/epoch_260_deploy.pth'

    from mmcv.runner import load_checkpoint,load_state_dict
    # 原模型读取
    config = mmcv.Config.fromfile(config_file)
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    load_checkpoint(model,checkpoint_file)
    
    deploy_model = repvgg_model_convert(model, save_checkpoint_file)
