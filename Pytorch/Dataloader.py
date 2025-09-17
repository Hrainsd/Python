# dataloader
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10("./dataset", train=True,
                                         transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, drop_last=False)
# shuffle 是否打乱数据
# drop_last 是否数量不足batch_size就舍去

img, target = train_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("DataLoader")
for episode in range(2):
    i = 0
    for data in train_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        # writer.add_images("Train_drop_last_true", imgs, i)
        # writer.add_images("Train_drop_last_false", imgs, i)
        writer.add_images("Dataset:{}".format(episode+1), imgs, i)
        i += 1
writer.close()
