# 网络模型的保存和读取
import torchvision

ResNet101 = torchvision.models.resnet101()

# 方法1：保存模型结构+模型参数
torch.save(ResNet101, "ResNet101.pth")

# 方法2：保存模型参数
torch.save(ResNet101.state_dict(), "ResNet101_parameters.pth")
