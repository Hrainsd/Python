# 现有模型的使用及修改---ResNet
import torchvision
from torch import nn

train_set = torchvision.datasets.ImageNet("./ImageNet_dataset", split="train",
                                          transform=torchvision.transforms.ToTensor())
resnet_False = torchvision.models.resnet101()
resnet_True = torchvision.models.resnet101(weights="DEFAULT")
print(resnet_False)
print("\n")
print(resnet_True)
print("\n")

resnet_False.fc = nn.Linear(2048, 10)
print(resnet_False)
print("\n")

# resnet_True.add_module("add_linear", nn.Linear(1000, 10))
# print(resnet_True)
# print("\n")
resnet_True.fc.add_module("add_linear", nn.Linear(1000, 10))
print(resnet_True)
