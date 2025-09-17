import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, SimpleRNN, LSTM, GRU
from keras.optimizers import Adam

# 参数设置
# 输入的长度
input_size = 28
# 序列的长度
time_size = 28
# 隐藏层cell的个数
num_cell = 50

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 格式-->>(-1, 28, 28)
# 数据预处理：归一化，one-hot编码
x_train = x_train/255.0
x_test = x_test/255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential([
    LSTM(units=num_cell, input_shape=(time_size, input_size)),
    Dense(units=10, activation='softmax')
])

adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 模型训练
epochs = 10
batch_size = 64
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# 模型评估
train_loss, train_accuracy = model.evaluate(x_train, y_train)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("train loss:{}, train accuracy:{}".format(train_loss, train_accuracy))
print("test loss:{}, test accuracy:{}".format(test_loss, test_accuracy))

# 可视化
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# 将真实的one-hot编码转换为类别标签
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
# 将预测结果转换为类别标签
y_train_pred = np.argmax(y_train_pred, axis=1)
y_test_pred = np.argmax(y_test_pred, axis=1)

plt.plot(y_train[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_train_pred[:500], label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Train(Accuracy:{:.4f}, Loss:{:.4f})'.format(train_accuracy, train_loss))
plt.show()

plt.plot(y_test[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_test_pred[:500], label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Test(Accuracy:{:.4f}, Loss:{:.4f})'.format(test_accuracy, test_loss))
plt.show()
