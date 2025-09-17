import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as calif
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as r2

# 加载数据集
data = calif()
x = data.data
y = data.target

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建模型
# loss为负值，是因为它是一种损失，sklearn给它加了负号，真正的loss是去掉负号后的值
# R方越接近1越好
lr = LR()
result = lr.fit(x_train, y_train)
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)
train_R2 = lr.score(x_train, y_train) # 计算结果是R方
test_R2 = lr.score(x_test, y_test)
train_MSE = MSE(y_train, y_pred_train)
test_MSE = MSE(y_test, y_pred_test)
cross_val = cross_val_score(lr, x, y, cv=10, scoring='neg_mean_squared_error')
R2_train = r2(y_train, y_pred_train)
R2_test = r2(y_test, y_pred_test)
print('Train MSE:{}, Test MSE:{}'.format(train_MSE, test_MSE))
print('Cross valldation score:{}'.format(cross_val))
print('Train R2:{}, Test R2:{}'.format(R2_train, R2_test))
print('截距项：{}'.format(lr.intercept_))
print([*zip(data.feature_names, lr.coef_)])
print('\n'.join('Feature name:{}, w:{}'.format(name, weight) for name, weight in zip(data.feature_names, lr.coef_)))

# 可视化
plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='', alpha=0.4)
plt.legend()
plt.title('Train(MSE:{:.4f}, R2:{:.4f})'.format(train_MSE, train_R2))
plt.show()

plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='', alpha=0.4)
plt.legend()
plt.title('Test(MSE:{:.4f}, R2:{:.4f})'.format(test_MSE, test_R2))
plt.show()

plt.plot(sorted(y_train), label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
plt.plot(sorted(y_pred_train), label='y_pred', marker='*', linestyle='', alpha=0.4)
plt.legend()
plt.title('Test(MSE:{:.4f}, R2:{:.4f})'.format(test_MSE, test_R2))
plt.show()

plt.plot(sorted(y_test), label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
plt.plot(sorted(y_pred_test), label='y_pred', marker='*', linestyle='', alpha=0.4)
plt.legend()
plt.title('Test(MSE:{:.4f}, R2:{:.4f})'.format(test_MSE, test_R2))
plt.show()
