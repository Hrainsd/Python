# Loss Function and backward
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torchvision
from torch.utils.data import DataLoader

input = torch.tensor([1, 2, 3], dtype=torch.float32)
output = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
output = torch.reshape(output, (1, 1, 1, 3))

Loss = L1Loss() # reduction默认为mean，也可以设置为sum
loss = Loss(input, output)
print(loss) # [(1-1)+(2-2)+(5-3)]/3

MseLoss = MSELoss()
mseloss = MseLoss(input, output)
print(mseloss)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
CrossLoss = CrossEntropyLoss()
crossloss = CrossLoss(x, y)
print(crossloss)

test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
data_loader = DataLoader(test_set, batch_size=64)

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
Cross_Loss = CrossEntropyLoss()

for data in data_loader:
    imgs, targets = data
    outputs = model(imgs)
    result_loss = Cross_Loss(outputs, targets)
    result_loss.backward()
    print(result_loss)
