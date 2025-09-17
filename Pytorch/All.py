# # 是否可用GPU
# import torch
# print(torch.cuda.is_available())
# 
# # 两大法宝函数
# # print(dir(torch))
# # help(torch)
# 
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import os
# 
# class MyData(Dataset):
# 
#     def __init__(self, root_dir, label_dir): __xx__是内置函数，不需要使用.xx()来调用
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(self.root_dir,self.label_dir)
#         self.img_path = os.listdir(self.path)
# 
#     def __getitem__(self, idx):
#         img_name = self.img_path[idx]
#         img_item_path = os.path.join(self.path,img_name)
#         img = Image.open(img_item_path)
#         label = self.label_dir
#         return img, label
# 
#     def __len__(self):
#         return len(self.img_path)
# 
# root_dir = "D:\\Photo"
# anime_label_dir = "动漫"
# anime_dataset = MyData(root_dir, anime_label_dir)
# 
# selfphoto_label_dir = "self_photo"
# selfphoto_dataset = MyData(root_dir, selfphoto_label_dir)
# train_dataset =  anime_dataset + selfphoto_dataset
# 
# print(len(anime_dataset))
# print(len(selfphoto_dataset))
# print(len(train_dataset))
# 
# img_1, label_1 = anime_dataset.__getitem__(728)
# img_1.show()
# img_2, label_2 = selfphoto_dataset[0]
# img_2.show()
# img_3, label_3 = train_dataset[728]
# img_3.show()
# img_4, label_4 = train_dataset[729]
# img_4.show()



# # TensorBoard
# # tensorboard --logdir="C:\Users\23991\OneDrive\桌面\Python\venv\Pytorch\logs"
#
# from PIL import Image
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("logs")
#
# for i in range(10):
#     writer.add_scalar("y = x^2", i^2, i)
#
# img_path = "D:\\Photo\\self_photo\\98d1de85da777748a04905a20fc208e.jpg"
# img_PIL = Image.open(img_path)
# img_array = np.array(img_PIL)
# print(img_array)
# print(img_array.shape) # 高度×宽度×通道数
# writer.add_image("photo1", img_array, 1, dataformats= "HWC") # 1表示第一幅图
# writer.add_image("photo1", img_array, 2, dataformats= "HWC")
# writer.add_image("photo2", img_array, 1, dataformats= "HWC")
# writer.close()



# # Transforms
# from torchvision import transforms
# from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
# import cv2
#
# img_path = "D:\\Photo\\self_photo\\98d1de85da777748a04905a20fc208e.jpg"
# img_PIL = Image.open(img_path) # PIL格式
#
# # 使用transforms
# # ToTensor
# to_tensor = transforms.ToTensor()
# img_tensor = to_tensor(img_PIL) # Tensor格式
# print(img_tensor)
#
# img_cv = cv2.imread(img_path) # ndarray格式
# print(img_cv)
#
# writer = SummaryWriter("logs")
# writer.add_image("photo", img_tensor)
# writer.close()



# # 常用的transforms
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
#
# img_PIL = Image.open("D:\\Photo\\动漫\\006yt1Omgy1guqyqguw6aj62ba4on1kz02.jpg")
# print(img_PIL)
#
# class Person:
#     def __call__(self, name):
#         print("Hello" + name)
#
#     def call2(self, name):
#         print("hello" + name)
#
# person = Person()
# person("张三")
# person.call2("李四")
#
# writer = SummaryWriter("logs")
# to_tensor = transforms.ToTensor()
# img_tensor = to_tensor(img_PIL)
# writer.add_image("ToTensor", img_tensor, 1)
# print(img_tensor.shape) # 通道数×高度×宽度
#
# # Normalize
# # 计算均值和标准差
# mean = torch.mean(img_tensor, dim=[1, 2])  # 按通道计算均值
# std = torch.std(img_tensor, dim=[1, 2])    # 按通道计算标准差
# print("均值: ", mean)
# print("标准差: ", std)
#
# print(img_tensor[0][0][0])
# trans_norm = transforms.Normalize(mean, std) # 均值，标准差
# img_norm = trans_norm(img_tensor)
# print(img_norm[0][0][0])
# writer.add_image("ToNorm", img_norm, 1)
#
# # Resize
# print(img_PIL.size)
# trans_resize = transforms.Resize((512, 512))
# img_resize = trans_resize(img_PIL)
# print(img_resize)
# img_resize = to_tensor(img_resize)
# writer.add_image("ToResize", img_resize, 1)
#
# # Compose
# trans_resize_2 = transforms.Resize(512) # 等比缩放
# trans_comp = transforms.Compose([trans_resize_2, to_tensor])
# img_comp = trans_comp(img_PIL)
# writer.add_image("ToCompose", img_comp, 1)
#
# # RandomCrop
# trans_random = transforms.RandomCrop(512) # 裁剪为正方形
# trans_comp2 = transforms.Compose([trans_random, to_tensor])
# for i in range(10):
#     img_RandomCrop = trans_comp2(img_PIL)
#     writer.add_image("ToRandomCrop", img_RandomCrop, i)
#
# writer.close()



# # torchvision & dataset
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
#
# transforms_dataset = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# train_set = torchvision.datasets.CIFAR10("./dataset", train=True, transform=transforms_dataset, download=True)
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False, transform=transforms_dataset, download=True)
#
# print(train_set[0])
# print(train_set.classes)
#
# img, target = train_set[0]
# print(img)
# print(target)
# print(train_set.classes[target])
#
# writer = SummaryWriter("DataLoader")
# for i in range(10):
#     img, target = train_set[i]
#     writer.add_image("Dataset", img, i)
#
# writer.close()



# # dataloader
# import torchvision
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# train_set = torchvision.datasets.CIFAR10("./dataset", train=True,
#                                          transform=torchvision.transforms.ToTensor(), download=True)
# train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, drop_last=False)
# # shuffle 是否打乱数据
# # drop_last 是否数量不足batch_size就舍去
#
# img, target = train_set[0]
# print(img.shape)
# print(target)
#
# writer = SummaryWriter("DataLoader")
#
# for episode in range(2):
#     i = 0
#     for data in train_loader:
#         imgs,targets = data
#         # print(imgs.shape)
#         # print(targets)
#         # writer.add_images("Train_drop_last_true", imgs, i)
#         # writer.add_images("Train_drop_last_false", imgs, i)
#         writer.add_images("Dataset:{}".format(episode+1), imgs, i)
#         i += 1
#
# writer.close()



# # nn.Module
# import torch
# from torch import nn
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input):
#         output = input + 1
#         return output
#
# model = Model()
# x = torch.tensor(1.0)
# y = model(x)
# print(x, y)



# # Conv
# import torch
# import torch.nn.functional as F
#
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]])
#
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
#
# input = torch.reshape(input, (1, 1, 5, 5))
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
# print(input.shape)
# print(kernel.shape)
#
# output1 = F.conv2d(input, kernel, stride=1)
# print(output1)
#
# output2 = F.conv2d(input, kernel, stride=1, padding=1)
# print(output2)



# # Conv2d
# import torch
# import torchvision
# from torch import nn
# from torch.nn import Conv2d
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
#                                         transform=torchvision.transforms.ToTensor(), download=True)
#
# data_loader = DataLoader(test_set, batch_size=64)
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
#
#     def forward(self, x):
#         y = self.conv1(x)
#         return y
#
# model = Model()
# print(model)
#
# writer = SummaryWriter("Conv2d")
# step = 1
# for data in data_loader:
#     imgs, targets = data
#     y = model(imgs)
#     print(imgs.shape)
#     print(y.shape)
#     writer.add_images("Input", imgs, step)
#     y = torch.reshape(y, (-1, 3, 30, 30)) # -1表示会自动计算维数
#     writer.add_images("Output", y, step)
#     step += 1
# writer.close()



# # MaxPool
# import torch
# from torch import nn
# from torch.nn import MaxPool2d
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
#
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
#                                         transform=torchvision.transforms.ToTensor(),download=True)
#
# data_loader = DataLoader(test_set, batch_size=64)
#
# input1 = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input1 = torch.reshape(input1, (-1, 1, 5, 5))
# print(input1.shape)
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)
#
#     def forward(self, input):
#         output = self.maxpool(input)
#         return output
#
# model = Model()
# output1 = model(input1)
# print(output1)
#
# writer = SummaryWriter("MaxPool")
# step = 1
#
# for data in data_loader:
#     imgs, targets = data
#     output = model(imgs)
#     writer.add_images("Before", imgs, step)
#     writer.add_images("After", output, step)
#     step += 1
#
# writer.close()



# # non-linear
# import torch
# import torchvision
# from torch import nn
# from torch.nn import ReLU
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
#                                         transform=torchvision.transforms.ToTensor(), download=True)
#
# data_loader = DataLoader(test_set, batch_size=64)
#
# input1 = torch.tensor([[1, -0.5],
#                       [-1, 3]])
#
# input1 = torch.reshape(input1, (-1, 1, 2, 2))
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.relu = ReLU()
#
#     def forward(self, input):
#         output = self.relu(input)
#         return output
#
# model = Model()
# output1 = model(input1)
# print(output1)
#
# step = 1
# writer = SummaryWriter("ReLu")
#
# for data in data_loader:
#     imgs, targets = data
#     output = model(imgs)
#     writer.add_images("Before", imgs, step)
#     writer.add_images("After", output, step)
#     step += 1
#
# writer.close()



# # Linear Layer and Norm Layer and other Layers
# import torch
# from torch import nn
# import torchvision
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn import Linear
#
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
#                                         transform=torchvision.transforms.ToTensor(), download=True)
#
# data_loader = DataLoader(test_set, batch_size=64, drop_last=True)
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear = Linear(196608, 10)
#
#     def forward(self, input):
#         output = self.linear(input)
#         return output
#
# model = Model()
# writer = SummaryWriter("Linear")
# step = 1
#
# for data in data_loader:
#     imgs, targets = data
#     output = torch.flatten(imgs)
#     print(imgs.shape)
#     print(output.shape)
#     output = model(output)
#     print(output.shape)
#     print(output)
#     writer.add_images("Before", imgs, step)
#     step += 1
#
# writer.close()



# # sequential
# import torch
# import torchvision
# from torch import nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
#
# # class Model(nn.Module):
# #     def __init__(self):
# #         super(Model, self).__init__()
# #         self.conv1 = Conv2d(3, 32, 5, padding=2)
# #         self.maxpool1 = MaxPool2d(2)
# #         self.conv2 = Conv2d(32, 32, 5, padding=2)
# #         self.maxpool2 = MaxPool2d(2)
# #         self.conv3 = Conv2d(32, 64, 5, padding=2)
# #         self.maxpool3 = MaxPool2d(2)
# #         self.flatten = Flatten()
# #         self.linear1 = Linear(1024, 64)
# #         self.linear2 = Linear(64, 10)
# #
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = self.maxpool1(x)
# #         x = self.conv2(x)
# #         x = self.maxpool2(x)
# #         x = self.conv3(x)
# #         x = self.maxpool3(x)
# #         x = self.flatten(x)
# #         x = self.linear1(x)
# #         x = self.linear2(x)
# #         return x
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
# model = Model()
# print(model)
#
# x = torch.ones((64, 3, 32, 32))
# y = model(x)
# print(y.shape)
#
# writer = SummaryWriter("Sequential")
# writer.add_graph(model, x)
# writer.close()



# # Loss Function and backward
# import torch
# from torch import nn
# from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
# import torchvision
# from torch.utils.data import DataLoader
#
# input = torch.tensor([1, 2, 3], dtype=torch.float32)
# output = torch.tensor([1, 2, 5], dtype=torch.float32)
#
# input = torch.reshape(input, (1, 1, 1, 3))
# output = torch.reshape(output, (1, 1, 1, 3))
#
# Loss = L1Loss() # reduction默认为mean，也可以设置为sum
# loss = Loss(input, output)
# print(loss) # [(1-1)+(2-2)+(5-3)]/3
#
# MseLoss = MSELoss()
# mseloss = MseLoss(input, output)
# print(mseloss)
#
# x = torch.tensor([0.1, 0.2, 0.3])
# y = torch.tensor([1])
# x = torch.reshape(x, (1, 3))
# CrossLoss = CrossEntropyLoss()
# crossloss = CrossLoss(x, y)
# print(crossloss)
#
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
#                                         transform=torchvision.transforms.ToTensor(), download=True)
#
# data_loader = DataLoader(test_set, batch_size=64)
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
# model = Model()
# Cross_Loss = CrossEntropyLoss()
#
# for data in data_loader:
#     imgs, targets = data
#     outputs = model(imgs)
#     result_loss = Cross_Loss(outputs, targets)
#     result_loss.backward()
#     print(result_loss)



# # optimizer
# import torch
# from torch import nn
# from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
# import torchvision
# from torch.utils.data import DataLoader
# from torch.optim import Adam
#
# input = torch.tensor([1, 2, 3], dtype=torch.float32)
# output = torch.tensor([1, 2, 5], dtype=torch.float32)
#
# input = torch.reshape(input, (1, 1, 1, 3))
# output = torch.reshape(output, (1, 1, 1, 3))
#
# Loss = L1Loss() # reduction默认为mean，也可以设置为sum
# loss = Loss(input, output)
# print(loss) # [(1-1)+(2-2)+(5-3)]/3
#
# MseLoss = MSELoss()
# mseloss = MseLoss(input, output)
# print(mseloss)
#
# x = torch.tensor([0.1, 0.2, 0.3])
# y = torch.tensor([1])
# x = torch.reshape(x, (1, 3))
# CrossLoss = CrossEntropyLoss()
# crossloss = CrossLoss(x, y)
# print(crossloss)
#
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
#                                         transform=torchvision.transforms.ToTensor(), download=True)
#
# data_loader = DataLoader(test_set, batch_size=64)
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
# model = Model()
# Cross_Loss = CrossEntropyLoss()
# optim = Adam(model.parameters(), lr=0.001)
# for episode in range(20):
#     for data in data_loader:
#         imgs, targets = data
#         outputs = model(imgs)
#         result_loss = Cross_Loss(outputs, targets)
#
#         optim.zero_grad()
#         result_loss.backward()
#         optim.step()
#     print(result_loss)



# # 现有模型的使用及修改 ResNet
# import torchvision
# from torch import nn
#
# # train_set = torchvision.datasets.ImageNet("./ImageNet_dataset", split="train",
# #                                           transform=torchvision.transforms.ToTensor())
#
# resnet_False = torchvision.models.resnet101()
# resnet_True = torchvision.models.resnet101(weights="DEFAULT")
# print(resnet_False)
# print("\n")
# print(resnet_True)
# print("\n")
#
# resnet_False.fc = nn.Linear(2048, 10)
# print(resnet_False)
# print("\n")
#
# # resnet_True.add_module("add_linear", nn.Linear(1000, 10))
# # print(resnet_True)
# # print("\n")
# resnet_True.fc.add_module("add_linear", nn.Linear(1000, 10))
# print(resnet_True)



# # 网络模型的保存和读取
# import torchvision
#
# ResNet101 = torchvision.models.resnet101()
#
# # 方法1：保存模型结构+模型参数
# torch.save(ResNet101, "ResNet101.pth")
#
# # 方法2：保存模型参数
# torch.save(ResNet101.state_dict(), "ResNet101_parameters.pth")


# # argmax
# import torch
#
# x = torch.tensor([[0.5, 1],
#                   [0.3, 2]])
#
# y1 = x.argmax(0) # 纵向从0开始，找最大值所在位置
# y2 = x.argmax(1) # 横向从0开始，找最大值所在位置
# print(y1)
# print(y2)



# # 完整模型训练
# import torch
# import torchvision
# from torch.utils.data import DataLoader
# from torch import nn
# from torch.utils.tensorboard import SummaryWriter
# import time
#
# # 创建数据集
# train_set = torchvision.datasets.CIFAR10("./dataset", train=True,
#                                          transform=torchvision.transforms.ToTensor(),download=True)
# test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
#                                          transform=torchvision.transforms.ToTensor(),download=True)
#
# # 数据集大小
# train_len = len(train_set)
# test_len = len(test_set)
#
# # 加载数据集
# train_loader = DataLoader(train_set, batch_size=64)
# test_loader = DataLoader(test_set, batch_size=64)
#
# # 定义训练的设备
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 创建模型
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = nn.Sequential(
#                 nn.Conv2d(3, 32, 5, padding=2),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(32, 32, 5, padding=2),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(32, 64, 5, padding=2),
#                 nn.MaxPool2d(2),
#                 nn.Flatten(),
#                 nn.Linear(1024, 64),
#                 nn.Linear(64, 10)
#                 )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
# model = Model()
# model.to(device)
#
# # 损失函数
# loss = nn.CrossEntropyLoss()
# loss.to(device)
#
# # 优化器
# learning_rate = 1e-3 # 0.01 1×10^(-2)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # 迭代次数
# epochs = 30
#
# # 步数
# total_train_step = 0
# total_test_step = 0
#
# # SummaryWriter
# writer = SummaryWriter("CompleteModel")
#
# start_time = time.time() # 记录训练开始时间
# for i in range(epochs):
#     print("--------第{}次迭代--------".format(i+1))
#     train_step = 0
#     test_step = 0
#     total_train_loss = 0
#     total_test_loss = 0
#     total_train_accuracy = 0
#     total_test_accuracy = 0
#
#     # 训练
#     model.train()
#     for data in train_loader:
#         imgs, targets = data
#         imgs = imgs.to(device)
#         targets = targets.to(device)
#         train_outputs = model(imgs)
#         train_loss = loss(train_outputs, targets)
#         total_train_loss += train_loss
#         total_train_accuracy += (train_outputs.argmax(1) == targets).sum()
#
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
#
#         train_step += 1 # 每次迭代中每步，迭代结束后从0开始
#         total_train_step += 1 # 所有迭代中每步
#         # if train_step % 100 == 0:
#         #     print("第{}次训练, 训练集损失：{}".format(train_step, total_train_loss.item()))
#
#     # 测试
#     model.eval()
#     with torch.no_grad():
#         for data in test_loader:
#             imgs, targets = data
#             imgs = imgs.to(device)
#             targets = targets.to(device)
#             test_outputs = model(imgs)
#             test_loss = loss(test_outputs, targets)
#             total_test_loss += test_loss
#             total_test_accuracy += (test_outputs.argmax(1) == targets).sum()
#
#             test_step += 1
#             total_test_step += 1
#             # if test_step % 100 == 0:
#             #     print("第{}次训练, 测试集损失：{}".format(test_step, total_test_loss.item()))
#     end_time = time.time()
#     print("第{}次迭代耗时：{}".format(i + 1, end_time-start_time))
#     print("训练集整体损失：{}".format(total_train_loss))
#     print("测试集整体损失：{}".format(total_test_loss))
#     print("训练集整体正确率：{}".format(total_train_accuracy/train_len))
#     print("测试集整体正确率：{}".format(total_test_accuracy/test_len))
#     writer.add_scalar("Train Loss", total_train_loss, total_train_step)
#     writer.add_scalar("Test Loss", total_test_loss, total_test_step)
#     writer.add_scalar("Train Accuracy", total_train_accuracy/train_len, total_train_step)
#     writer.add_scalar("Test Accuracy", total_test_accuracy/test_len, total_test_step)
#
# torch.save(model, "Model.pth")
# print("模型已保存至Model.pth")
# writer.close()



# # 模型验证
# import torch
# import torchvision
# from torch import nn
# from PIL import Image
#
# # 数据
# img_path = "C:\\Users\\23991\\Downloads\\airplane.jpg" # 飞机的图片
# img_PIL = Image.open(img_path)
# img_tensor = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
#                                              torchvision.transforms.ToTensor()])
# img = img_tensor(img_PIL)
# img = torch.reshape(img, (1, 3, 32, 32))
#
# # 创建模型
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = nn.Sequential(
#                 nn.Conv2d(3, 32, 5, padding=2),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(32, 32, 5, padding=2),
#                 nn.MaxPool2d(2),
#                 nn.Conv2d(32, 64, 5, padding=2),
#                 nn.MaxPool2d(2),
#                 nn.Flatten(),
#                 nn.Linear(1024, 64),
#                 nn.Linear(64, 10)
#                 )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
# model = torch.load("Model.pth")
#
# model.eval()
# with torch.no_grad():
#     output = model(img)
#     print(output)
#
# print("预测类别：{}".format((output.argmax(1)).item()))
