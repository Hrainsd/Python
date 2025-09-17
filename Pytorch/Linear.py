# Linear Layer and Norm Layer and other Layers
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear

test_set = torchvision.datasets.CIFAR10("./dataset", train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
data_loader = DataLoader(test_set, batch_size=64, drop_last=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        output = self.linear(input)
        return output

model = Model()
writer = SummaryWriter("Linear")
step = 1

for data in data_loader:
    imgs, targets = data
    output = torch.flatten(imgs)
    print(imgs.shape)
    print(output.shape)
    output = model(output)
    print(output.shape)
    print(output)
    writer.add_images("Before", imgs, step)
    writer.add_images("After", output, step)
    step += 1
writer.close()
