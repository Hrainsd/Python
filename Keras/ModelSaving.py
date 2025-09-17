import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

# 数据预处理
# 展平图像数据并归一化
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
print(x_train.shape)
print(x_test.shape)
# label使用ont hot编码
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(y_train.shape)
print(y_test.shape)

# 优化器
sgd = SGD(lr=0.01)
adam = Adam(lr=0.001)

# 创建模型
model = Sequential([
    Dense(units=10, activation='softmax'),
])
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 参数设置
batch_size = 64
epochs = 10

# 模型训练
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# 模型评估
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print("train loss:", train_loss)
print("train accuracy:", train_accuracy)

loss, accuracy = model.evaluate(x_test, y_test)
print("test loss:", loss)
print("test accuracy:", accuracy)

# 保存模型
model.save("model.h5") # HDF5文件
