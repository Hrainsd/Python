import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\SVM\1.1.csv"
df = pd.read_csv(file_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建 SVM 模型并调整参数
model = SVR(kernel='rbf', C=20, gamma='auto', epsilon=0.1)

# 训练模型
model.fit(X_train_scaled, y_train)

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

# 计算训练集的 RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train_scaled)))

# 绘制训练集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.title(f'训练集预测结果对比 \nRMSE: {rmse_train:.2f}')
plt.plot(y_train, label='真实值', color='#E0BBE4')
plt.plot(model.predict(X_train_scaled), marker='o', label='预测值', color='#FF8D00')
plt.legend(frameon=False, bbox_to_anchor=(1.15, 1))
plt.xlabel('样本序号')
plt.ylabel('标签值')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体
plt.savefig('Train.svg', format='svg', bbox_inches='tight')
plt.show()

# 计算训练集的 RMSE
rmse_test = np.sqrt(mean_squared_error(y_test, predictions))

# 绘制测试集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.title(f'测试集预测结果对比 \nRMSE: {rmse_test:.2f}')
plt.plot(y_test, label='真实值', color='#E0BBE4')
plt.plot(predictions, marker='o', label='预测值', color='#FF8D00')
plt.legend(frameon=False, bbox_to_anchor=(1.15, 1))
plt.xlabel('样本序号')
plt.ylabel('标签值')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体
plt.savefig('Test.svg', format='svg', bbox_inches='tight')
plt.show()
