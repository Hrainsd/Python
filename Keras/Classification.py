import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Adagrad, rmsprop, Adadelta
from keras.regularizers import l2

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
    Dense(units=200, input_dim=784, activation='tanh', kernel_regularizer=l2(l2=0.0003)),
    Dropout(0.4),
    Dense(units=100, activation='tanh', kernel_regularizer=l2(l2=0.0003)),
    Dropout(0.4),
    Dense(units=10, activation='softmax', kernel_regularizer=l2(l2=0.003)),
])
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 参数设置
batch_size = 64
epochs = 30

# 模型训练
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# 模型评估
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print("train loss:", train_loss)
print("train accuracy:", train_accuracy)

loss, accuracy = model.evaluate(x_test, y_test)
print("test loss:", loss)
print("test accuracy:", accuracy)

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
