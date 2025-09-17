import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 加载CSV数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\DecisionTree\1.csv"
df = pd.read_csv(file_path)

# 准备数据：拆分特征（X）和目标变量（y）
X = df.iloc[:, :-1]  # 所有列除了最后一列是特征
y = df.iloc[:, -1]   # 最后一列是目标变量

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建标准化对象
scaler = StandardScaler()

# 拟合并转换训练集特征
X_train_scaled = scaler.fit_transform(X_train)

# 仅转换测试集特征
X_test_scaled = scaler.transform(X_test)

# 创建并训练决策树回归模型
model = DecisionTreeRegressor()
model.fit(X_train_scaled, y_train)

# 进行预测
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 计算训练集和测试集的均方根误差
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# 计算训练集和测试集的平均绝对误差
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# 计算训练集和测试集的平均相对误差
def mean_relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_train = mean_relative_error(y_train, y_train_pred)
mape_test = mean_relative_error(y_test, y_test_pred)

print("Training Set:")
print(f"MSE: {mse_train:.2f}")
print(f"MAE: {mae_train:.2f}")
print(f"MAPE: {mape_train:.2f}%")

print("\nTesting Set:")
print(f"MSE: {mse_test:.2f}")
print(f"MAE: {mae_test:.2f}")
print(f"MAPE: {mape_test:.2f}%")

# 计算训练集和测试集的均方根误差
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# 绘制训练集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.plot(y_train.values, label='Actual', color='#E4C1F9')
plt.plot(y_train_pred, label='Predicted', color='#66CCCC')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title(f'Training Set Prediction Comparison \nRMSE: {rmse_train:.2f}')
plt.legend(frameon=False)
plt.show()

# 绘制测试集预测结果对比曲线图
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', color='#E4C1F9')
plt.plot(y_test_pred, label='Predicted', color='#66CCCC')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title(f'Testing Set Prediction Comparison \nRMSE: {rmse_test:.2f}')
plt.legend(frameon=False)
plt.show()
