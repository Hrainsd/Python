import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体

# 读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\CNN\1.csv"
df = pd.read_csv(file_path)

# 划分特征和标签
X = df.iloc[:, :-1].values  # 特征
y = df.iloc[:, -1].values   # 标签

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 调整输入数据形状以符合CNN的要求（这里假设每个特征是一个时间步）
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 构建CNN模型
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=2000, batch_size=64, verbose=1)

# 使用模型预测训练集和测试集上的结果
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算均方误差（MSE）
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 计算均方根误差（RMSE）
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

# 计算平均绝对误差（MAE）
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# 计算平均偏差误差（MBE）
train_mbe = np.mean(y_train - y_train_pred)
test_mbe = np.mean(y_test - y_test_pred)

# 输出结果
print("训练集 MSE:", train_mse)
print("测试集 MSE:", test_mse)
print("训练集 RMSE:", train_rmse)
print("测试集 RMSE:", test_rmse)
print("训练集 MAE:", train_mae)
print("测试集 MAE:", test_mae)
print("训练集 MBE:", train_mbe)
print("测试集 MBE:", test_mbe)

# 绘制训练集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.plot(y_train, label='实际值', color='#A8DADC')
plt.plot(y_train_pred, label='预测值', color='#E0BBE4')
plt.title(f'训练集预测结果对比 \nRMSE: {train_rmse:.2f}')
plt.xlabel('样本编号')
plt.ylabel('值')
plt.legend()
plt.grid(True)
plt.show()

# 绘制测试集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='实际值', color='#A8DADC')
plt.plot(y_test_pred, label='预测值', color='#E0BBE4')
plt.title(f'测试集预测结果对比 \nRMSE: {test_rmse:.2f}')
plt.xlabel('样本编号')
plt.ylabel('值')
plt.legend()
plt.grid(True)
plt.show()
