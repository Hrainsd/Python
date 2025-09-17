import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
input_data = pd.read_csv(r"D:\大学\本科\计算机\人工智能\机器学习\深度学习框架--Tensorflow\data.csv")

# 数据预处理
# ont-hot编码
input_data = pd.get_dummies(input_data)

features = input_data.drop('actual', axis=1)
labels = input_data['actual']
features = np.array(features)
labels = np.array(labels)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 将数据重新调整为 [样本数, 时间步长, 特征数] 的形状
# 假设时间步长为1
time_step = 1
x_train = np.reshape(x_train, (x_train.shape[0], time_step, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], time_step, x_test.shape[1]))

# 参数设置
# 输入的长度
input_size = x_train.shape[2]
# 序列的长度
time_size = time_step
# 隐藏层cell的个数
num_cell = 20

# 优化器
learning_rate = 1e-3
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=num_cell, input_shape=(time_size, input_size),activation='relu', return_sequences=True, kernel_initializer='random_normal'),
    tf.keras.layers.LSTM(units=num_cell, activation='relu', kernel_initializer='random_normal'),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer='l2')
])
model.compile(optimizer=optimizer, loss='mse')

# 模型训练
batch_size = 32
epochs = 1000
results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# 模型评估
train_loss = model.evaluate(x_train, y_train)
test_loss = model.evaluate(x_test, y_test)
print("train loss:{}, test loss:{}".format(train_loss, test_loss))

# 结果可视化
# 损失曲线
plt.plot(results.history['loss'])
plt.title('Train loss curve')
plt.show()

# 预测曲线
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
plt.plot(y_train, label='y_train_true')
plt.plot(y_train_pred, label='y_train_pred')
plt.legend()
plt.title('Train(loss:{:.4f})'.format(train_loss))
plt.show()

plt.plot(y_test, label='y_test_true')
plt.plot(y_test_pred, label='y_test_pred')
plt.legend()
plt.title('Test(loss:{:.4f})'.format(test_loss))
plt.show()
