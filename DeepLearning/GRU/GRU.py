import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout

# 指定使用宋体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载数据集
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\LSTM\1.csv"
df = pd.read_csv(file_path)

# 将数据转换为 numpy 数组
data = df.values.astype('float32')

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 将数据拆分为训练集和测试集
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

# 将值数组转换为数据集矩阵
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 50
X_train, Y_train = create_dataset(train, time_step)
X_test, Y_test = create_dataset(test, time_step)

# 重新整形输入以匹配 GRU 模型的需求
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 构建 GRU 模型
model = Sequential()
model.add(GRU(units=128, return_sequences=True, input_shape=(1, time_step)))
model.add(GRU(units=128))
model.add(Dense(64, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, Y_train, epochs=300, batch_size=50, verbose=2)

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化预测值
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# 计算 RMSE 评估模型性能
train_score = np.sqrt(np.mean(np.square(train_predict - Y_train)))
test_score = np.sqrt(np.mean(np.square(test_predict - Y_test)))

# 输出结果
train_mse = np.mean((train_predict - Y_train) ** 2)
test_mse = np.mean((test_predict - Y_test) ** 2)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = np.mean(np.abs(train_predict - Y_train))
test_mae = np.mean(np.abs(test_predict - Y_test))
train_mbe = np.mean((train_predict - Y_train) / Y_train)
test_mbe = np.mean((test_predict - Y_test) / Y_test)

print("训练集 MSE:", train_mse)
print("测试集 MSE:", test_mse)
print("训练集 RMSE:", train_rmse)
print("测试集 RMSE:", test_rmse)
print("训练集 MAE:", train_mae)
print("测试集 MAE:", test_mae)
print("训练集 MBE:", train_mbe)
print("测试集 MBE:", test_mbe)

# 绘制训练集预测结果对比曲线图
plt.figure(figsize=(10, 6))
plt.plot(Y_train.flatten(), label='Actual', color='#A8DADC')
plt.plot(train_predict.flatten(), label='Predicted', color='#A4D4AE')
plt.title(f'训练集预测结果对比 \nRMSE: {train_rmse:.2f}')
plt.xlabel('样本')
plt.ylabel('值')
plt.legend()
plt.show()

# 绘制测试集预测结果对比曲线图
plt.figure(figsize=(10, 6))
plt.plot(Y_test.flatten(), label='Actual', color='#A8DADC')
plt.plot(test_predict.flatten(), label='Predicted', color='#A4D4AE')
plt.title(f'测试集预测结果对比 \nRMSE: {test_rmse:.2f}')
plt.xlabel('样本')
plt.ylabel('值')
plt.legend()
plt.show()
