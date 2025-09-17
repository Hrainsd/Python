# non-linear
import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
data_loader = DataLoader(test_set, batch_size=64)

input1 = torch.tensor([[1, -0.5],
                      [-1, 3]])

input1 = torch.reshape(input1, (-1, 1, 2, 2))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = ReLU()

    def forward(self, input):
        output = self.relu(input)
        return output

model = Model()
output1 = model(input1)
print(output1)

step = 1
writer = SummaryWriter("ReLu")

for data in data_loader:
    imgs, targets = data
    output = model(imgs)
    writer.add_images("Before", imgs, step)
    writer.add_images("After", output, step)
    step += 1
writer.close()
