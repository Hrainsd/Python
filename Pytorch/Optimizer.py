# optimizer
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam

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
optim = Adam(model.parameters(), lr=0.001)
for episode in range(20):
    for data in data_loader:
        imgs, targets = data
        outputs = model(imgs)
        result_loss = Cross_Loss(outputs, targets)

        optim.zero_grad()
        result_loss.backward()
        optim.step()
    print(result_loss)
