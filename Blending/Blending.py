# Blending ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_california_housing as calif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor as xgbreg
from sklearn.metrics import mean_squared_error as mse

# 加载数据集
data = calif()
x = data.data
y = data.target

# 标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 数据预处理
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建模型并训练
reg_svm = SVR(kernel='rbf')
reg_rf = RandomForestRegressor(n_estimators=100)
reg_xgb = xgbreg(n_estimators=100, learning_rate=0.1)

result_svm = reg_svm.fit(x_train, y_train)
result_rf = reg_rf.fit(x_train, y_train)
result_xgb = reg_xgb.fit(x_train, y_train)

y_train_pred_svm = reg_svm.predict(x_train)
y_test_pred_svm = reg_svm.predict(x_test)
y_train_pred_rf = reg_rf.predict(x_train)
y_test_pred_rf = reg_rf.predict(x_test)
y_train_pred_xgb = reg_xgb.predict(x_train)
y_test_pred_xgb = reg_xgb.predict(x_test)

R2_svm_train = reg_svm.score(x_train, y_train)
R2_svm_test = reg_svm.score(x_test, y_test)
R2_rf_train = reg_rf.score(x_train, y_train)
R2_rf_test = reg_rf.score(x_test, y_test)
R2_xgb_train = reg_xgb.score(x_train, y_train)
R2_xgb_test = reg_xgb.score(x_test, y_test)

MSE_svm_train = mse(y_train, y_train_pred_svm)
MSE_svm_test = mse(y_test, y_test_pred_svm)
MSE_rf_train = mse(y_train, y_train_pred_rf)
MSE_rf_test = mse(y_test, y_test_pred_rf)
MSE_xgb_train = mse(y_train, y_train_pred_xgb)
MSE_xgb_test = mse(y_test, y_test_pred_xgb)

print('Train:\nSVM: R2:{}, MSE:{}\nRF: R2:{}, MSE:{}\nXGBoost: R2:{}, MSE:{}'.format(
    R2_svm_train, MSE_svm_train, R2_rf_train, MSE_rf_train, R2_xgb_train, MSE_xgb_train))
print('Test:\nSVM: R2:{}, MSE:{}\nRF: R2:{}, MSE:{}\nXGBoost: R2:{}, MSE:{}'.format(
    R2_svm_test, MSE_svm_test, R2_rf_test, MSE_rf_test, R2_xgb_test, MSE_xgb_test))

# 融合模型
x_train_blend = np.vstack((y_train_pred_svm, y_train_pred_rf, y_train_pred_xgb)).T
x_test_blend = np.vstack((y_test_pred_svm, y_test_pred_rf, y_test_pred_xgb)).T

reg_linear = LinearRegression()
result = reg_linear.fit(x_train_blend, y_train)

coef = reg_linear.coef_
b = reg_linear.intercept_

y_train_pred_blend = reg_linear.predict(x_train_blend)
y_test_pred_blend = reg_linear.predict(x_test_blend)

R2_blend_train = reg_linear.score(x_train_blend, y_train)
R2_blend_test = reg_linear.score(x_test_blend, y_test)

MSE_blend_train = mse(y_train, y_train_pred_blend)
MSE_blend_test = mse(y_test, y_test_pred_blend)

print('Blended Model Coefficients:', coef)
print('Blended Model Intercept:', b)
print('Blended Model Train: R2: {}, MSE: {}'.format(R2_blend_train, MSE_blend_train))
print('Blended Model Test: R2: {}, MSE: {}'.format(R2_blend_test, MSE_blend_test))

# 可视化
plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_train_pred_blend, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Train Set: R2:{:.4f}, MSE:{:.4f}'.format(R2_blend_train, MSE_blend_train))
plt.show()

plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_test_pred_blend, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Test Set: R2:{:.4f}, MSE:{:.4f}'.format(R2_blend_test, MSE_blend_test))
plt.show()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred_blend, color='blue', label='Predicted', alpha=0.6)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', label='Ideal', lw=2)
plt.title('Train Set: True vs Predicted (Blended Model)')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_blend, color='green', label='Predicted', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal', lw=2)
plt.title('Test Set: True vs Predicted (Blended Model)')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.show()
