# 完整模型训练
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

# 创建数据集
train_set = torchvision.datasets.CIFAR10("./dataset", train=True,
                                         transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# 数据集大小
train_len = len(train_set)
test_len = len(test_set)

# 加载数据集
train_loader = DataLoader(train_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

# 定义训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(3, 32, 5, padding=2),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding=2),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5, padding=2),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1024, 64),
                nn.Linear(64, 10)
                )

    def forward(self, x):
        x = self.model(x)
        return x

model = Model()
model.to(device)

# 损失函数
loss = nn.CrossEntropyLoss()
loss.to(device)

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 迭代次数
epochs = 30

# 步数
total_train_step = 0
total_test_step = 0

# SummaryWriter
writer = SummaryWriter("CompleteModel")

start_time = time.time() # 记录训练开始时间
for i in range(epochs):
    print("--------第{}次迭代--------".format(i+1))
    train_step = 0
    test_step = 0
    total_train_loss = 0
    total_test_loss = 0
    total_train_accuracy = 0
    total_test_accuracy = 0

    # 训练
    model.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        train_outputs = model(imgs)
        train_loss = loss(train_outputs, targets)
        total_train_loss += train_loss
        total_train_accuracy += (train_outputs.argmax(1) == targets).sum()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_step += 1 # 每次迭代中每步，迭代结束后从0开始
        total_train_step += 1 # 所有迭代中每步
        # if train_step % 100 == 0:
        #     print("第{}次训练, 训练集损失：{}".format(train_step, total_train_loss.item()))

    # 测试
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            test_outputs = model(imgs)
            test_loss = loss(test_outputs, targets)
            total_test_loss += test_loss
            total_test_accuracy += (test_outputs.argmax(1) == targets).sum()

            test_step += 1
            total_test_step += 1
            # if test_step % 100 == 0:
            #     print("第{}次训练, 测试集损失：{}".format(test_step, total_test_loss.item()))
    end_time = time.time()
    print("第{}次迭代耗时：{}".format(i + 1, end_time-start_time))
    print("训练集整体损失：{}".format(total_train_loss))
    print("测试集整体损失：{}".format(total_test_loss))
    print("训练集整体正确率：{}".format(total_train_accuracy/train_len))
    print("测试集整体正确率：{}".format(total_test_accuracy/test_len))
    writer.add_scalar("Train Loss", total_train_loss, total_train_step)
    writer.add_scalar("Test Loss", total_test_loss, total_test_step)
    writer.add_scalar("Train Accuracy", total_train_accuracy/train_len, total_train_step)
    writer.add_scalar("Test Accuracy", total_test_accuracy/test_len, total_test_step)

torch.save(model, "Model.pth")
print("模型已保存至Model.pth")
writer.close()

# 模型验证
import torch
import torchvision
from torch import nn
from PIL import Image

# 数据
img_path = "C:\\Users\\23991\\Downloads\\airplane.jpg" # 验证图片
img_PIL = Image.open(img_path)
img_tensor = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor()])
img = img_tensor(img_PIL)
img = torch.reshape(img, (1, 3, 32, 32))

# 创建模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(3, 32, 5, padding=2),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 5, padding=2),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5, padding=2),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1024, 64),
                nn.Linear(64, 10)
                )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("Model.pth")
model.eval()
with torch.no_grad():
    output = model(img)
    print(output)
print("预测类别：{}".format((output.argmax(1)).item()))
