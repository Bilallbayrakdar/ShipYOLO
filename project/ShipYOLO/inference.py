import torch
import torchvision.transforms as transforms

import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from models.models import *
from resize import *
# from utils.datasets import LoadStreams, LoadImages

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transforms_img = transforms.Compose([
    # transforms.Resize([416, 416]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
half = device.type != 'cpu'

# change model to deploy model ".deploy"
def change_model(weights = "runs-new/best.pth", cfg = "cfg/yolov4-RepVGGUnit-RFB-FPN-cbam.cfg", img_size = 416):
    
    train_model = Darknet(cfg, img_size).to(device)
    # print(model)
    checkpoint = torch.load(weights, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
    train_model.load_state_dict(ckpt)

    deploy_model = repvgg_model_convert(train_model, Darknet(cfg, img_size), save_path='runs-new/yolo-my.pth')

def process_image(img, min_side):
    size = img.shape
    h, w = size[0], size[1]
    #长边缩放为min_side 
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    print(top, bottom, left, right)
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
    #print pad_img.shape
    #cv2.imwrite("after-" + os.path.basename(filename), pad_img)

    return pad_img, 

def plot_(img):
    # img = Image.open(os.path.join('images', '2007_000648' + '.jpg'))

    plt.figure("Image") # 图像窗口名称
    plt.imshow(img)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show()

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # new_unpad = new_shape
    # print("new_unpad:", new_unpad)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def resize_img_keep_ratio(img_name,target_size):
    img = cv2.imread(img_name)
    old_size= img.shape[0:2]
    #ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv2.resize(img,(new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
    return img_new, img

def read_once_img(img_size, image_path):
    img0 = cv2.imread(image_path)
    img = letterbox(img0, new_shape=img_size)[0]
    # plot_(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # print(img.shape)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
    return img, img0


def inference_image(model, img_size, image_path="inference/input/0057.jpg"):
    # img, img0 = read_once_img(img_size, image_path)
    # print("1", img.shape)
    img0 = cv2.imread(image_path)
    img = letterbox(img0, new_shape=img_size)[0]
    print(img.shape, img0.shape)

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)

    # img, img0 = resize_img_keep_ratio(image_path, [img_size, img_size])
    # plot_(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # # Convert
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # img = np.ascontiguousarray(img)
    # print(img.shape)
    # img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    # img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # img = img.unsqueeze(0)
    # print("2", img.shape)


    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5, agnostic=False)  # 一个类别在一个里面

    # for i, det in enumerate(pred):


    for i, det in enumerate(pred):
        # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        det = det.detach().cpu().numpy()
        print(det)
        for bbox in det:
            cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
    
    cv2.imwrite("1.jpg", img0)
    plot_img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    plot_(plot_img)


def main(cfg, img_size, weights):
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    img0, bbox_list = inference_image(model, img_size, image_path="inference/input/0132.jpg")

    draw_bbox(img, bbox_list)

if __name__ == "__main__":
    # train_model()
    # test()
    # inference_image()
#     change_model()
    weights = "runs/yolov4-last-416.pth"
    cfg = "cfg/yolov4.cfg"
    main(cfg, 320, weights)