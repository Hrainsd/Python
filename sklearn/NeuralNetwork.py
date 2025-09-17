import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor as MLPreg
from sklearn.datasets import fetch_california_housing as calif
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = calif()
x = data.data
y = data.target

# 数据预处理---标准化
stdscaler = StandardScaler()
x = stdscaler.fit_transform(x)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建模型
# activation must be a str among {'relu', 'tanh', 'identity', 'logistic'}
reg = MLPreg(hidden_layer_sizes=(32, 64, 128), activation='relu', solver='adam')
result = reg.fit(x_train, y_train)
y_pred_train = reg.predict(x_train)
y_pred_test = reg.predict(x_test)
score_train = reg.score(x_train, y_train)
score_test = reg.score(x_test, y_test)
MSE_train = MSE(y_train, y_pred_train)
MSE_test = MSE(y_test, y_pred_test)
print('Train:R2:{}, MSE:{}'.format(score_train, MSE_train))
print('Test:R2:{}, MSE:{}'.format(score_test, MSE_test))

# 可视化
plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Train(R2:{:.4f}, MSE:{:.4f})'.format(score_train, MSE_train))
plt.show()

plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Test(R2:{:.4f}, MSE:{:.4f})'.format(score_test, MSE_test))
plt.show()
