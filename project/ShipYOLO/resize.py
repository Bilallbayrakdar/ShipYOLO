import cv2
import matplotlib.pyplot as plt

def plot_(img):
    # img = Image.open(os.path.join('images', '2007_000648' + '.jpg'))

    plt.figure("Image") # 图像窗口名称
    plt.imshow(img)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show()

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

if __name__ == "__main__":
  img = "inference/input/0056.jpg"
  target_size = [416, 416]
  img_new, img0 = resize_img_keep_ratio(img, target_size)
  plot_img = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
  print(img_new.shape)

  plot_(plot_img)
