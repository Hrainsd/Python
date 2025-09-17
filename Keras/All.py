# # 线性回归
#
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense
#
# # 输入
# x = np.random.rand(100)
# noise = np.random.normal(0, 0.01, x.shape)
#
# # 输出
# y = 0.1 * x + 0.2 + noise
#
# # 可视化：散点图
# plt.scatter(x, y)
# plt.show()
#
# # 创建模型
# model = Sequential()
# model.add(Dense(units=1, input_dim=1))
# model.compile(optimizer="adam", loss="mse")
#
# # 训练
# epochs = 5000
# for i in range(epochs):
#     cost = model.train_on_batch(x, y)
#     if (i + 1) % 100 == 0:
#         print("cost:{}".format(cost))
#
# # 打印模型参数
# W, b = model.layers[0].get_weights()
# print("W:{}, b:{}".format(W, b))
#
# # 模型预测
# pred_y = model.predict(x)
#
# # 可视化模型结果
# plt.scatter(x, y)
# plt.plot(x, pred_y, "r-", lw=3)
# plt.show()



# # 非线性回归
#
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.optimizers import Adam
#
# # 载入数据
# x = np.linspace(-2, 2, num=200)
# noise = np.random.normal(0, 0.5, x.shape)
#
# y = np.square(x) + noise
#
# plt.scatter(x, y)
# plt.show()
#
# # 优化器
# optim = Adam(lr=0.01)
#
# # 创建模型
# model = Sequential()
# model.add(Dense(units=10, input_dim=1, activation='tanh'))
# # model.add(Activation('tanh'))
# model.add(Dense(units=1))
# # model.add(Activation('tanh'))
# model.compile(optimizer=optim, loss='mse')
#
# # 参数设置
# epochs = 3000
#
# # 模型训练
# for i in range(epochs):
#     cost = model.train_on_batch(x, y)
#     if (i + 1) % 100 == 0:
#         print("cost:", cost)
#
# # 预测
# pred_y = model.predict(x)
#
# # 可视化
# plt.scatter(x, y)
# plt.plot(x, pred_y, 'c-', lw=3)
# plt.show()



# # 分类
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD, Adam, Adagrad, rmsprop, Adadelta
# from keras.regularizers import l2
#
# # 载入数据
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# print(x_train.shape)
# print(y_train.shape)
#
# # 数据预处理
# # 展平图像数据并归一化
# x_train = x_train.reshape(x_train.shape[0], -1)/255.0
# x_test = x_test.reshape(x_test.shape[0], -1)/255.0
# print(x_train.shape)
# print(x_test.shape)
# # label使用ont hot编码
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
# print(y_train.shape)
# print(y_test.shape)
#
# # 优化器
# sgd = SGD(lr=0.01)
# adam = Adam(lr=0.001)
#
# # 创建模型
# model = Sequential([
#     Dense(units=200, input_dim=784, activation='tanh', kernel_regularizer=l2(l2=0.0003)),
#     Dropout(0.4),
#     Dense(units=100, activation='tanh', kernel_regularizer=l2(l2=0.0003)),
#     Dropout(0.4),
#     Dense(units=10, activation='softmax', kernel_regularizer=l2(l2=0.003)),
# ])
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 参数设置
# batch_size = 64
# epochs = 30
#
# # 模型训练
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# print("train loss:", train_loss)
# print("train accuracy:", train_accuracy)
#
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("test loss:", test_loss)
# print("test accuracy:", test_accuracy)
#
# # 可视化
# y_train_pred = model.predict(x_train)
# y_test_pred = model.predict(x_test)
#
# # 将真实的one-hot编码转换为类别标签
# y_train = np.argmax(y_train, axis=1)
# y_test = np.argmax(y_test, axis=1)
# # 将预测结果转换为类别标签
# y_train_pred = np.argmax(y_train_pred, axis=1)
# y_test_pred = np.argmax(y_test_pred, axis=1)
#
# plt.plot(y_train[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_train_pred[:500], label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(Accuracy:{:.4f}, Loss:{:.4f})'.format(train_accuracy, train_loss))
# plt.show()
#
# plt.plot(y_test[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_test_pred[:500], label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(Accuracy:{:.4f}, Loss:{:.4f})'.format(test_accuracy, test_loss))
# plt.show()



# # 卷积神经网络CNN---手写数字识别
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
# from keras.optimizers import Adam
#
# # 加载数据集
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # 数据预处理
# x_train = x_train.reshape(-1, 28, 28, 1)/255.0
# x_test = x_test.reshape(-1, 28, 28, 1)/255.0
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
#
# # 创建模型
# model = Sequential([
#     Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding='same', activation='relu'),
#     MaxPool2D(pool_size=2, strides=2, padding='same'),
#     Conv2D(64, 5, 1, 'same', activation='relu'),
#     MaxPool2D(2, 2, 'same'),
#     Flatten(),
#     Dense(units=1024, activation='relu'),
#     Dropout(0.5),
#     Dense(units=10, activation='softmax')
# ])
#
# adam = Adam(learning_rate=0.0001)
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 模型训练
# epochs = 10
# batch_size = 64
#
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("train loss:{}, train accuracy:{}".format(train_loss, train_accuracy))
# print("test loss:{}, test accuracy:{}".format(test_loss, test_accuracy))
#
# # 可视化
# y_train_pred = model.predict(x_train)
# y_test_pred = model.predict(x_test)
#
# # 将真实的one-hot编码转换为类别标签
# y_train = np.argmax(y_train, axis=1)
# y_test = np.argmax(y_test, axis=1)
# # 将预测结果转换为类别标签
# y_train_pred = np.argmax(y_train_pred, axis=1)
# y_test_pred = np.argmax(y_test_pred, axis=1)
#
# plt.plot(y_train[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_train_pred[:500], label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(Accuracy:{:.4f}, Loss:{:.4f})'.format(train_accuracy, train_loss))
# plt.show()
#
# plt.plot(y_test[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_test_pred[:500], label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(Accuracy:{:.4f}, Loss:{:.4f})'.format(test_accuracy, test_loss))
# plt.show()


# # 循环神经网络RNN
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, SimpleRNN, LSTM, GRU
# from keras.optimizers import Adam
#
# # 参数设置
# # 输入的长度
# input_size = 28
# # 序列的长度
# time_size = 28
# # 隐藏层cell的个数
# num_cell = 50
#
# # 加载数据集
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # 格式-->>(-1, 28, 28)
# # 数据预处理：归一化，one-hot编码
# x_train = x_train/255.0
# x_test = x_test/255.0
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
#
# # 创建模型
# model = Sequential([
#     LSTM(units=num_cell, input_shape=(time_size, input_size)),
#     Dense(units=10, activation='softmax')
# ])
#
# adam = Adam(learning_rate=0.0001)
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 模型训练
# epochs = 10
# batch_size = 64
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("train loss:{}, train accuracy:{}".format(train_loss, train_accuracy))
# print("test loss:{}, test accuracy:{}".format(test_loss, test_accuracy))
#
# # 可视化
# y_train_pred = model.predict(x_train)
# y_test_pred = model.predict(x_test)
#
# # 将真实的one-hot编码转换为类别标签
# y_train = np.argmax(y_train, axis=1)
# y_test = np.argmax(y_test, axis=1)
# # 将预测结果转换为类别标签
# y_train_pred = np.argmax(y_train_pred, axis=1)
# y_test_pred = np.argmax(y_test_pred, axis=1)
#
# plt.plot(y_train[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_train_pred[:500], label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(Accuracy:{:.4f}, Loss:{:.4f})'.format(train_accuracy, train_loss))
# plt.show()
#
# plt.plot(y_test[:500], label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_test_pred[:500], label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(Accuracy:{:.4f}, Loss:{:.4f})'.format(test_accuracy, test_loss))
# plt.show()


# # 模型保存
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD, Adam
#
# # 载入数据
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# print(x_train.shape)
# print(y_train.shape)
#
# # 数据预处理
# # 展平图像数据并归一化
# x_train = x_train.reshape(x_train.shape[0], -1)/255.0
# x_test = x_test.reshape(x_test.shape[0], -1)/255.0
# print(x_train.shape)
# print(x_test.shape)
# # label使用ont hot编码
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
# print(y_train.shape)
# print(y_test.shape)
#
# # 优化器
# sgd = SGD(lr=0.01)
# adam = Adam(lr=0.001)
#
# # 创建模型
# model = Sequential([
#     Dense(units=10, activation='softmax'),
# ])
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 参数设置
# batch_size = 64
# epochs = 10
#
# # 模型训练
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# print("train loss:", train_loss)
# print("train accuracy:", train_accuracy)
#
# loss, accuracy = model.evaluate(x_test, y_test)
# print("test loss:", loss)
# print("test accuracy:", accuracy)
#
# # 保存模型
# model.save("model.h5") # HDF5文件



# # 模型加载
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from keras.models import load_model
# from keras.utils import np_utils
#
# # 载入数据
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # 数据预处理
# # 展平图像数据并归一化
# x_train = x_train.reshape(x_train.shape[0], -1)/255.0
# x_test = x_test.reshape(x_test.shape[0], -1)/255.0
# # label使用ont hot编码
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
#
# # 加载模型
# model = load_model("model.h5")
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# print("train loss:", train_loss)
# print("train accuracy:", train_accuracy)
#
# loss, accuracy = model.evaluate(x_test, y_test)
# print("test loss:", loss)
# print("test accuracy:", accuracy)



# # 仅保存和加载网络参数、网络结构
# # 仅保存参数和加载参数
# model.save_weights("model_weights.h5")
# model.load_weights("model_weights.h5")
#
# # 仅保存网络结构和加载网络结构
# from keras.models import model_from_json
# json_string = model.to_json()
# load_model = model_from_json(json_string)



# # 绘制网络结构
# # 卷积神经网络CNN---手写数字识别
# import keras
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
# from keras.optimizers import Adam
# from keras.utils.vis_utils import plot_model
#
# # 加载数据集
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # 数据预处理
# x_train = x_train.reshape(-1, 28, 28, 1)/255.0
# x_test = x_test.reshape(-1, 28, 28, 1)/255.0
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
#
# # 创建模型
# model = Sequential([
#     Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding='same', activation='relu'),
#     MaxPool2D(pool_size=2, strides=2, padding='same'),
#     Conv2D(64, 5, 1, 'same', activation='relu'),
#     MaxPool2D(2, 2, 'same'),
#     Flatten(),
#     Dense(units=1024, activation='relu'),
#     Dropout(0.5),
#     Dense(units=10, activation='softmax')
# ])
#
# # 绘制网络结构
# plot_model(model, "model.png", show_shapes=True, show_layer_names=True, rankdir="TB")
# plt.figure(figsize=(10, 10))
# img = plt.imread("model.png")
# plt.imshow(img)
# plt.axis("off")
# plt.show()
