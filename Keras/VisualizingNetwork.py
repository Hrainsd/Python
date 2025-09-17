# 绘制网络结构
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, 28, 28, 1)/255.0
x_test = x_test.reshape(-1, 28, 28, 1)/255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential([
    Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding='same', activation='relu'),
    MaxPool2D(pool_size=2, strides=2, padding='same'),
    Conv2D(64, 5, 1, 'same', activation='relu'),
    MaxPool2D(2, 2, 'same'),
    Flatten(),
    Dense(units=1024, activation='relu'),
    Dropout(0.5),
    Dense(units=10, activation='softmax')
])

# 绘制网络结构
plot_model(model, "model.png", show_shapes=True, show_layer_names=True, rankdir="TB")
plt.figure(figsize=(10, 10))
img = plt.imread("model.png")
plt.imshow(img)
plt.axis("off")
plt.show()
