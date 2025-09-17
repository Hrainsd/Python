# torchvision & dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter

transforms_dataset = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10("./dataset", train=True, transform=transforms_dataset, download=True)
test_set = torchvision.datasets.CIFAR10("./dataset", train=False, transform=transforms_dataset, download=True)

print(train_set[0])
print(train_set.classes)

img, target = train_set[0]
print(img)
print(target)
print(train_set.classes[target])

writer = SummaryWriter("DataLoader")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("Dataset", img, i)
writer.close()
