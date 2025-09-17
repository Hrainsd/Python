import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
# 展平图像数据并归一化
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
# label使用ont hot编码
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 加载模型
model = load_model("model.h5")

# 模型评估
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print("train loss:", train_loss)
print("train accuracy:", train_accuracy)

loss, accuracy = model.evaluate(x_test, y_test)
print("test loss:", loss)
print("test accuracy:", accuracy)

# 仅保存参数和加载参数
model.save_weights("model_weights.h5")
model.load_weights("model_weights.h5")

# 仅保存网络结构和加载网络结构
from keras.models import model_from_json
json_string = model.to_json()
load_model = model_from_json(json_string)
