# # 常量、变量
# import tensorflow as tf
# import numpy as np
#
# print(tf.__version__)
#
# # 创建常量op
# op1 = tf.constant([[2, 2]])
# op2 = tf.constant([[1], [2]])
#
# # 创建矩阵乘法op
# product = tf.matmul(op1, op2)
# print(product)
#
# # 直接输出结果
# print(product.numpy())  # TensorFlow 2.x 可以直接通过 .numpy() 获取张量的值
#
# # 创建一个变量op
# x = tf.Variable([[6, 6]])
# c = tf.constant([[1, 2]])
#
# # 创建加法、减法op
# add = tf.add(x, c)
# sub = tf.subtract(x, c)
# print(add.numpy())
# print(sub.numpy())
#
# # 变量赋值
# state = tf.Variable([0])
# for i in range(5):
#     operation = tf.add(state, [1])
#     state.assign(operation)
#     print(state.numpy())
# print(state)
#
# # 转换格式
# x = tf.constant([[1, 2], [3, 4]])
# x.numpy()
# print(x.numpy())
#
# y = tf.cast(x, dtype=tf.float32)
# print(y)
#
# z = np.array([1, 6])
#
# p = tf.multiply(z, 2)
# print(p)



# # 线性回归
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# # 数据
# x = np.random.rand(100)
# noise = np.random.normal(0, 0.02, size=x.shape)
# y = 0.5 * x + 0.6 + noise
#
# plt.scatter(x, y)
# plt.show()
#
# # 创建模型变量
# k = tf.Variable(0.0)
# b = tf.Variable(0.0)
#
# # 损失函数
# def compute_loss():
#     y_hat = k * x + b
#     return tf.reduce_mean(tf.square(y_hat - y))
#
# # 优化器
# optimizer = tf.optimizers.Adam(learning_rate=0.001)
#
# # 模型训练
# epochs = 3000
# for i in range(epochs):
#     optimizer.minimize(compute_loss, var_list=[k, b])
#
#     # 每100次迭代打印一次结果
#     if (i + 1) % 100 == 0:
#         loss = compute_loss().numpy()
#         print("Iteration{}, loss: {:.4f}, k: {:.4f}, b: {:.4f}".format(i + 1, loss, k.numpy(), b.numpy()))
#
# # 可视化结果
# cmap = plt.cm.rainbow
# y_pred = k.numpy() * x + b.numpy()
# plt.scatter(x, y, cmap=cmap)
# plt.plot(x, y_pred, 'c-', label='Fitted Line', lw=3)
# plt.legend()
# plt.show()



# # 非线性回归
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# # 数据
# x = np.linspace(-1, 1, 200).reshape(-1, 1)
# noise = np.random.normal(0, 0.02, x.shape)
# y = np.square(x) + noise
#
# plt.scatter(x, y)
# plt.title("Original Data")
# plt.show()
#
# # 数据预处理
# x = tf.convert_to_tensor(x, dtype=tf.float32)
# y = tf.convert_to_tensor(y, dtype=tf.float32)
#
# # 创建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=10, activation='tanh'),
#     tf.keras.layers.Dense(units=1, activation='linear')
# ])
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 损失函数
# def loss_fn(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))
#
# # 训练模型
# epochs = 3000
# for epoch in range(epochs):
#     with tf.GradientTape() as tape:
#         y_pred = model(x)
#         loss = loss_fn(y, y_pred)
#
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")
#
# # 可视化结果
# y_pred = model(x).numpy()
# plt.scatter(x, y, label='Data')
# plt.plot(x, y_pred, 'c-', label='Fitted Line', lw=3)
# plt.title("Fitted Results")
# plt.legend()
# plt.show()



# # 分类
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 数据集
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # 数据预处理
# x_train = x_train.reshape(-1, 784)/255.0
# x_test = x_test.reshape(-1, 784)/255.0
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 创建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=128, input_dim=784, activation='relu'),
#     tf.keras.layers.Dense(units=10, activation='softmax')
# ])
# model.compile(optimizer=optimizer,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # 模型训练
# epochs = 20
# batch_size = 64
# results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("train loss:{}, train accuracy:{}".format(train_loss, train_accuracy))
# print("test loss:{}, test accuracy:{}".format(test_loss, test_accuracy))
#
# # 可视化
# # 损失曲线
# plt.plot(results.history["loss"], label="Train Loss")
# plt.plot(results.history["val_loss"], label="Test Loss")
# plt.legend()
# plt.title("Loss curve")
# plt.show()
#
# # 准确率
# plt.plot(results.history["accuracy"], label="Train Accuracy")
# plt.plot(results.history["val_accuracy"], label="Test Accuracy")
# plt.legend()
# plt.title("Accuracy curve")
# plt.show()
#
# # 对比图
# y_test_pred = model.predict(x_test)
# plt.plot(y_test[:200], label="y_true", marker='o', markerfacecolor='none') # 颜色设置为透明
# plt.plot(y_test_pred[:200], label='y_pred', marker='x')
# plt.legend()
# plt.title("Test")
# plt.show()



# # 回归任务
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.keras
# from tensorflow.keras import layers
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
#
# # 加载数据
# input_data = pd.read_csv("D:\大学\本科\计算机\人工智能\机器学习\深度学习框架--Tensorflow\data.csv")
# print(input_data.head())
# print(input_data.shape)
#
# # 处理时间数据
# year = input_data['year']
# month = input_data['month']
# day = input_data['day']
#
# # datetime格式
# dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day))
#          for year, month, day in zip(year, month, day)]
# dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
#
# # 特征数据可视化
# plt.plot(input_data['temp_2'], label="temp_2", color='c', linestyle='-')
# plt.plot(input_data['temp_1'], label="temp_1", color='pink', linestyle='--')
# plt.plot(input_data['average'], label="average", linestyle='-.')
# plt.plot(input_data['friend'], label="friend", linestyle=':')
# plt.legend()
# plt.title("Feature Curves")
# plt.show()
#
# # one-hot编码
# input_data = pd.get_dummies(input_data)
# print(input_data.head())
#
# # 定义特征和标签
# features = input_data.drop('actual', axis=1)
# labels = input_data['actual']
#
# # 将特征转换为 NumPy 数组
# features = np.array(features)
# labels = np.array(labels)
#
# # 按照 8:2 比例划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
# # 特征标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 创建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=128, activation='relu', input_dim=14, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(rate=0.5),
#     tf.keras.layers.Dense(units=1, activation='relu', kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
# ])
# model.compile(optimizer=optimizer, loss='mse')
#
# # 模型训练
# result = model.fit(X_train, y_train, batch_size=64, epochs=1000)
#
# # 模型评估
# train_loss = model.evaluate(X_train, y_train)
# test_loss = model.evaluate(X_test, y_test)
# print("train_loss:{}, test loss:{}".format(train_loss, test_loss))
#
# # 模型结构可视化
# model.summary()
#
# # 结果可视化
# # 模型预测
# y_train_pred = model.predict(X_train)
# plt.plot(y_train_pred, label="y_pred", color='c')
# plt.plot(y_train, label="y_true", color='pink')
# plt.legend()
# plt.title("Train loss curve(Loss:{})".format(train_loss))
# plt.show()
#
# y_test_pred = model.predict(X_test)
# plt.plot(y_test_pred, label="y_pred", color='c')
# plt.plot(y_test, label="y_true", color='pink')
# plt.legend()
# plt.title("Test loss curve(Loss:{})".format(test_loss))
# plt.show()



# # 分类
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# # 加载数据
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # 数据预处理
# x_train = x_train.reshape(-1, 784)/255.0
# x_test = x_test.reshape(-1, 784)/255.0
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 创建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, input_dim=784, activation='tanh'),
#     tf.keras.layers.Dense(units=10, activation='softmax')
# ])
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 模型训练
# batch_size = 64
# epochs = 30
# results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print("train loss:{}, train accuracy:{}".format(train_loss, train_accuracy))
# print("test loss:{}, test accuracy:{}".format(test_loss, test_accuracy))
#
# # 结果可视化
# # 损失曲线
# plt.plot(results.history["loss"])
# plt.title("Train loss curve")
# plt.show()
#
# # 准确率
# plt.plot(results.history["accuracy"])
# plt.title("Train accuracy")
# plt.show()
#
# # 对比图
# y_test_pred = model.predict(x_test)
# plt.plot(y_test[:100], label="y_test", marker='o', markerfacecolor='none')
# plt.plot(y_test_pred[:100], label="y_test_pred", marker='x')
# plt.legend()
# plt.title("Test")
# plt.show()



# # 模型的保存和读取
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# # 加载数据
# fasion_mnist = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = fasion_mnist.load_data()
#
# # 数据预处理
# x_train = x_train.reshape(-1, 784)/255.0
# x_test = x_test.reshape(-1, 784)/255.0
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 创建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, input_dim=784, activation='tanh'),
#     tf.keras.layers.Dense(units=10, activation='softmax')
# ])
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 模型训练
# batch_size = 64
# epochs = 30
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
# # 模型评估
# train_loss, train_accuracy = model.evaluate(x_train, y_train)
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print(f"train loss:{train_loss}, train accuracy:{train_accuracy}")
# print(f"test loss:{test_loss}, test accuracy:{test_accuracy}")
#
# # 结果可视化
# # 对比图
# y_test_pred = model.predict(x_test)
# plt.plot(y_test[:100], label="y_test", marker='o', markerfacecolor='none')
# plt.plot(y_test_pred[:100], label='y_test_pred', marker='x')
# plt.legend()
# plt.title("Test")
# plt.show()
#
# # 模型保存（保存网络结构和参数）
# model.save("model.h5")
# # 模型读取
# model = tf.keras.models.load_model("model.h5")
#
# # 仅保存网络结构
# config = model.to_json()
# # 模型结构读取
# model = tf.keras.models.model_from_json(config)
# with open("config.json", 'w') as json:
#     json.write(config)
# model.summary()
#
# # 仅保存网络参数
# weights = model.get_weights()
# model.save_weights("model_weights.h5")
# # 模型参数读取
# model.load_weights("model_weights.h5")



# # 卷积神经网络---猫狗识别
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#
# # 定义图像大小和批次大小
# img_height, img_width = 64, 64
# batch_size = 64
#
# # 数据集目录
# data_dir = r"D:\大学\本科\计算机\人工智能\机器学习\深度学习框架--Tensorflow\dogs-vs-cats\dataset"
#
# # 数据预处理
# # 使用 ImageDataGenerator 对图像进行实时数据增强
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     validation_split=0.2,
#     rotation_range=45,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode="nearest"
# )
# test_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     validation_split=0.2
# )
#
# # 生成训练数据
# train_data = train_datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training',
# )
# # 生成验证数据
# val_data = test_datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation'
# )
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 创建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, 5, activation='relu', input_shape=(img_height, img_width, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#
#     tf.keras.layers.Conv2D(64, 5, activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2')
# ])
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
#
# # 模型训练
# epochs = 20
# result = model.fit(train_data, epochs=epochs, validation_data=val_data)
#
# # 结果可视化
# # 损失曲线
# plt.plot(result.history['loss'], label='Train loss')
# plt.plot(result.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.title('Loss curves')
# plt.show()
#
# # 准确率曲线
# plt.plot(result.history['acc'], label='Train accuracy')
# plt.plot(result.history['val_acc'], label='Validation accuracy')
# plt.legend()
# plt.title('Accuracy curves')
# plt.show()



# # 迁移学习---猫狗识别
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
#
# # 定义图像大小和批次大小
# img_height, img_width = 64, 64
# batch_size = 64
#
# # 数据集目录
# data_dir = r"D:\大学\本科\计算机\人工智能\机器学习\深度学习框架--Tensorflow\dogs-vs-cats\dataset"
#
# # 数据预处理
# # 使用 ImageDataGenerator 对图像进行实时数据增强
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     validation_split=0.2,
# )
# val_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     validation_split=0.2
# )
#
# # 生成训练数据
# train_data = train_datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training',
# )
# # 生成验证数据
# val_data = val_datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation'
# )
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 创建模型
# model = tf.keras.applications.VGG16(include_top=False,
#                                         weights='imagenet',
#                                         input_shape=(img_height, img_width, 3))
# # 冻结预训练模型的参数
# model.trainable = False
# # 自定义全连接层
# x = model.output
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(units=128, activation='relu',, kernel_regularizer='l2')(x)
# x = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)
#
# model = tf.keras.models.Model(model.input, x)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
#
# # 模型训练
# epochs = 10
# callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# result = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[callbacks])
#
# # 结果可视化
# # 损失曲线
# plt.plot(result.history['loss'], label='Train loss')
# plt.plot(result.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.title('Loss curves')
# plt.show()
#
# # 准确率曲线
# plt.plot(result.history['acc'], label='Train accuracy')
# plt.plot(result.history['val_acc'], label='Validation accuracy')
# plt.legend()
# plt.title('Accuracy curves')
# plt.show()



# # 循环神经网络RNN
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# # 加载数据
# input_data = pd.read_csv(r"D:\大学\本科\计算机\人工智能\机器学习\深度学习框架--Tensorflow\data.csv")
#
# # 数据预处理
# # ont-hot编码
# input_data = pd.get_dummies(input_data)
#
# features = input_data.drop('actual', axis=1)
# labels = input_data['actual']
# features = np.array(features)
# labels = np.array(labels)
#
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
# # 标准化
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#
# # 将数据重新调整为 [样本数, 时间步长, 特征数] 的形状
# # 假设时间步长为1
# time_step = 1
# x_train = np.reshape(x_train, (x_train.shape[0], time_step, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], time_step, x_test.shape[1]))
#
# # 参数设置
# # 输入的长度
# input_size = x_train.shape[2]
# # 序列的长度
# time_size = time_step
# # 隐藏层cell的个数
# num_cell = 20
#
# # 优化器
# learning_rate = 1e-3
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
# # 创建模型
# model = tf.keras.Sequential([
#     tf.keras.layers.SimpleRNN(units=num_cell, input_shape=(time_size, input_size),activation='relu', return_sequences=True, kernel_initializer='random_normal'),
#     tf.keras.layers.LSTM(units=num_cell, activation='relu', kernel_initializer='random_normal'),
#     tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer='l2'),
#     tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer='l2')
# ])
# model.compile(optimizer=optimizer, loss='mse')
#
# # 模型训练
# batch_size = 32
# epochs = 1000
# results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
# # 模型评估
# train_loss = model.evaluate(x_train, y_train)
# test_loss = model.evaluate(x_test, y_test)
# print("train loss:{}, test loss:{}".format(train_loss, test_loss))
#
# # 结果可视化
# # 损失曲线
# plt.plot(results.history['loss'])
# plt.title('Train loss curve')
# plt.show()
#
# # 预测曲线
# y_train_pred = model.predict(x_train)
# y_test_pred = model.predict(x_test)
# plt.plot(y_train, label='y_train_true')
# plt.plot(y_train_pred, label='y_train_pred')
# plt.legend()
# plt.title('Train(loss:{:.4f})'.format(train_loss))
# plt.show()
#
# plt.plot(y_test, label='y_test_true')
# plt.plot(y_test_pred, label='y_test_pred')
# plt.legend()
# plt.title('Test(loss:{:.4f})'.format(test_loss))
# plt.show()
