# sequential
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = Conv2d(3, 32, 5, padding=2)
#         self.maxpool1 = MaxPool2d(2)
#         self.conv2 = Conv2d(32, 32, 5, padding=2)
#         self.maxpool2 = MaxPool2d(2)
#         self.conv3 = Conv2d(32, 64, 5, padding=2)
#         self.maxpool3 = MaxPool2d(2)
#         self.flatten = Flatten()
#         self.linear1 = Linear(1024, 64)
#         self.linear2 = Linear(64, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.maxpool2(x)
#         x = self.conv3(x)
#         x = self.maxpool3(x)
#         x = self.flatten(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = Model()
print(model)

x = torch.ones((64, 3, 32, 32))
y = model(x)
print(y.shape)

writer = SummaryWriter("Sequential")
writer.add_graph(model, x)
writer.close()
