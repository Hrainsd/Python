# Transforms
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2

img_path = "D:\\Photo\\self_photo\\98d1de85da777748a04905a20fc208e.jpg"
img_PIL = Image.open(img_path) # PIL格式

# 使用transforms
# ToTensor
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_PIL) # Tensor格式
print(img_tensor)

img_cv = cv2.imread(img_path) # ndarray格式
print(img_cv)

writer = SummaryWriter("logs")
writer.add_image("photo", img_tensor)
writer.close()



# 常用的transforms
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_PIL = Image.open("D:\\Photo\\动漫\\006yt1Omgy1guqyqguw6aj62ba4on1kz02.jpg")
print(img_PIL)

class Person:
    def __call__(self, name):
        print("Hello" + name)

    def call2(self, name):
        print("hello" + name)

person = Person()
person("张三")
person.call2("李四")

writer = SummaryWriter("logs")
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_PIL)
writer.add_image("ToTensor", img_tensor, 1)
print(img_tensor.shape) # 通道数×高度×宽度

# Normalize
# 计算均值和标准差
mean = torch.mean(img_tensor, dim=[1, 2])  # 按通道计算均值
std = torch.std(img_tensor, dim=[1, 2])    # 按通道计算标准差
print("均值: ", mean)
print("标准差: ", std)

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(mean, std) # 均值，标准差
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("ToNorm", img_norm, 1)

# Resize
print(img_PIL.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_PIL)
print(img_resize)
img_resize = to_tensor(img_resize)
writer.add_image("ToResize", img_resize, 1)

# Compose
trans_resize_2 = transforms.Resize(512) # 等比缩放
trans_comp = transforms.Compose([trans_resize_2, to_tensor])
img_comp = trans_comp(img_PIL)
writer.add_image("ToCompose", img_comp, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512) # 裁剪为正方形
trans_comp2 = transforms.Compose([trans_random, to_tensor])
for i in range(10):
    img_RandomCrop = trans_comp2(img_PIL)
    writer.add_image("ToRandomCrop", img_RandomCrop, i)
writer.close()
