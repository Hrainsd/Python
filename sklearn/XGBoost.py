import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor as xgb
from sklearn.datasets import fetch_california_housing as calif
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error as MSE

# 加载数据集
data = calif()
x = data.data
y = data.target
print('Data shape:{}'.format(x.shape))

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建模型
reg = xgb(n_estimators=162, learning_rate=0.1)
result = reg.fit(x_train, y_train)
y_pred_train = reg.predict(x_train)
y_pred_test = reg.predict(x_test)
score_train = reg.score(x_train, y_train) # R方
score_test = reg.score(x_test, y_test)
mse_train = MSE(y_train, y_pred_train)
mse_test = MSE(y_test, y_pred_test)
feature_importances = reg.feature_importances_

# 交叉验证的结果和score的结果是相同类型的，比如回归问题默认都是R2，分类问题默认都是准确率
cross_val_R2 = cross_val_score(reg, x_train, y_train, cv=10).mean()
cross_val_MSE = cross_val_score(reg, x_train, y_train, scoring='neg_mean_squared_error', cv=10).mean()
print(*zip(data.feature_names, feature_importances))
print('Cross validation score:R2:{:.4f}, MSE:{:.4f}'.format(cross_val_R2, cross_val_MSE))
kfold = KFold(n_splits=10, shuffle=False)
cross_val_R2_kfold = cross_val_score(reg, x_train, y_train, cv=kfold).mean()
print('Cross validation score:KFold:{:.4f}'.format(cross_val_R2_kfold))

# 可视化
# 学习曲线
R2_values = []
MSE_values = []
for i in range(1, 501, 1):
    reg = xgb(n_estimators=i)
    result = reg.fit(x_train, y_train)
    y_pred_test = reg.predict(x_test)
    R2_values.append(reg.score(x_test, y_test))
    MSE_values.append(MSE(y_test, y_pred_test))
print('n_estimator:{}, Max R2:{}'.format(R2_values.index(max(R2_values)), max(R2_values)))
print('n_estimator:{}, Min MSE:{}'.format(MSE_values.index(min(MSE_values)), min(MSE_values)))
plt.plot(R2_values, label='R2', marker='o')
plt.plot(MSE_values, label='MSE', marker='o')
plt.title('Learning curve')
plt.show()

plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Train(R2:{:.4f}, MSE:{:.4f})'.format(score_train, mse_train))
plt.show()

plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Test(R2:{:.4f}, MSE:{:.4f})'.format(score_test, mse_test))
plt.show()
