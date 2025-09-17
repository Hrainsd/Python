import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# 读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\NeuralNetwork\1.1.csv"
df = pd.read_csv(file_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 定义优化器并设置学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

# 编译模型
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 训练模型
history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=64, validation_split=0.2)

# 使用模型进行预测
predictions = model.predict(X_test_scaled)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# 计算平均相对误差
mean_relative_error = np.mean(np.abs((y_test - predictions) / y_test))
print("Mean Relative Error:", mean_relative_error)

# 计算R2分数
r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)

# 绘制代价函数随迭代次数的曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='训练集')
plt.plot(history.history['val_loss'], label='验证集')
plt.title('代价函数随迭代次数的曲线')
plt.xlabel('迭代次数')
plt.ylabel('代价函数值')
plt.legend()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体，设置一次就好，不用设置很多次
plt.savefig('neural_network_loss.svg', format='svg', bbox_inches='tight')
plt.show()

# 计算训练集的 RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train_scaled)))

# 绘制训练集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.title(f'训练集预测结果对比 \nRMSE: {rmse_train:.2f}')
plt.plot(y_train, label='真实值', color='#FF6347')
plt.plot(model.predict(X_train_scaled), marker='o', label='预测值', color='#9AD9CA')
plt.legend(frameon=False, bbox_to_anchor=(1.15, 1))
plt.xlabel('样本序号')
plt.ylabel('标签值')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体
plt.savefig('neural_network_train.svg', format='svg', bbox_inches='tight')
plt.show()

# 计算训练集的 RMSE
rmse_test = np.sqrt(mean_squared_error(y_test, predictions))

# 绘制测试集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.title(f'测试集预测结果对比 \nRMSE: {rmse_test:.2f}')
plt.plot(y_test, label='真实值', color='#FF6347')
plt.plot(predictions, marker='o', label='预测值', color='#9AD9CA')
plt.legend(frameon=False, bbox_to_anchor=(1.15, 1))
plt.xlabel('样本序号')
plt.ylabel('标签值')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体
plt.savefig('neural_network_test.svg', format='svg', bbox_inches='tight')
plt.show()
