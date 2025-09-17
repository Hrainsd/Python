# TensorBoard
# tensorboard --logdir="C:\Users\23991\OneDrive\桌面\Python\venv\Pytorch\logs"

from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")

for i in range(10):
    writer.add_scalar("y = x^2", i^2, i)

img_path = "D:\\Photo\\self_photo\\98d1de85da777748a04905a20fc208e.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(img_array)
print(img_array.shape) # 高度×宽度×通道数
writer.add_image("photo1", img_array, 1, dataformats= "HWC") # 1表示第一幅图
writer.add_image("photo1", img_array, 2, dataformats= "HWC")
writer.add_image("photo2", img_array, 1, dataformats= "HWC")
writer.close()
