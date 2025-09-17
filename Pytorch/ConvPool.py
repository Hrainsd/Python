# Conv
import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input.shape)
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
print(output1)

output2 = F.conv2d(input, kernel, stride=1, padding=1)
print(output2)



# Conv2d
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
data_loader = DataLoader(test_set, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        return y

model = Model()
print(model)
writer = SummaryWriter("Conv2d")
step = 1

for data in data_loader:
    imgs, targets = data
    y = model(imgs)
    print(imgs.shape)
    print(y.shape)
    writer.add_images("Input", imgs, step)
    y = torch.reshape(y, (-1, 3, 30, 30)) # -1表示会自动计算维数
    writer.add_images("Output", y, step)
    step += 1
writer.close()



# MaxPool
import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)
data_loader = DataLoader(test_set, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output

model = Model()
writer = SummaryWriter("MaxPool")
step = 1

for data in data_loader:
    imgs, targets = data
    output = model(imgs)
    writer.add_images("Before", imgs, step)
    writer.add_images("After", output, step)
    step += 1
writer.close()
