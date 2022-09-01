Paper: ShipYOLO: An Enhanced Model for Ship Detection

DOI: https://doi.org/10.1155/2021/1060182

Xu Han, "ShipYOLO: An Enhanced Model for Ship Detection," Journal of Advanced Transportation, vol. 2021, Article ID 1060182, 11 pages, 2021. https://doi.org/10.1155/2021/1060182.

# Bibtex
```bibtex
@article{
    author    = {Xu Han and Lining Zhao and Yue Ning and Jingfeng Hu}, 
    title     = {{ShipYOLO}: An Enhanced Model for Ship Detection}, 
    journal   = {Journal of Advanced Transportation},
    month     = {jun},  
    year      = {2021},
    pages     = {1--11}
    }
```

*completed*
- ShipYOLO
- Results for WSODD.

*ongoing*
- ShipYOLOv2
- Results for Seaships, SMD, et al.

# Results and Models

Inference in NVIDIA GeForce RTX 3060 Laptop GPU.

## Dataset
- WSODD
  
  DownLoad Link:[https://github.com/sunjiaen/WSODD](https://github.com/sunjiaen/WSODD)

  Train list and Test list for this paper:[train.txt](datasets/WSODD/ImageSets/train.txt),[test.txt](datasets/WSODD/ImageSets/test.txt)

  The number of images in "datasets/WSODD" is only to give an example.


| Model | Size | $mAP_{coco}$ | $mAP_{coco}@50$ | $mAP_{coco}@75$ | FPS | Config | Datasets | Download |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv4 | 512 | 0.251| 0.582 | 0.173 | 34.50 | mmdet | WSODD | ------ |
| ShipYOLO | 512 | 0.296 | 0.615 | 0.237 | 44.96 | mmdet | WSODD | ------ |
| ShipYOLOV2 | 512 | 0.296 | 0.617 | 0.241 | 44.20 | mmdet | WSODD | ------ |
| - | 512 | 0.423 | 0.780 | 0.391 | 44.20 | mosaic+aug | WSODD | ------ |

# 1. Installation related environment
pip install -r requirements.txt

# 2. Train
```shell
bash run_train.sh

python re_pth.py
```

# 3. demo
```python
import torch
from models.shipyolo.shipyolo import ShipYOLOv2
from utils.general import non_max_suppression

'''
Train deploy=False
Test  deploy=True
'''
x = torch.rand(1, 3, 288, 512).cuda()
model = ShipYOLOv2(num_classes=14, deploy=True).cuda()
output = model(x)  # without nms
output = non_max_suppression(output[0], conf_thres=0.001, iou_thres=0.5)
```

# 4. structure
The old version of the code(ShipYOLO) is in "project/ShipYOLO".

We also built the code(ShipYOLO-mmdet) based on mmdetection which is in "project/ShipYOLO-mmdet".

## Acknowledgements
- [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
- [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

## Communication

- zhaolining@dlmu.edu.cn
- cn.xuhan@139.com