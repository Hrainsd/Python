import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# 加载数据集
train_data = pd.read_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\House Prices - Advanced Regression Techniques\train.csv')
x_test = pd.read_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\House Prices - Advanced Regression Techniques\test.csv')

train_id = train_data['Id']
test_id = x_test['Id']
y_train = train_data['SalePrice']

x_train = train_data.drop(columns=['SalePrice', 'Id'])
x_test = x_test.drop(columns=['Id'])

# 数据预处理---特征工程
# 合并数据
x_data = pd.concat((x_train, x_test), axis=0)
x_data['MSSubClass'] = x_data['MSSubClass'].astype(str)

# 非数值数据进行ont-hot编码
x_data = pd.get_dummies(x_data)
print('One-Hot编码后的特征：\n', x_data.head())

# 数值数据进行标准化
numeric_cols = x_data.select_dtypes(include=[np.number]).columns
print('数值数据的列表：', numeric_cols)
std = StandardScaler()
x_data[numeric_cols] = std.fit_transform(x_data[numeric_cols])

# 数值数据填补缺失值
print('缺失值列表：\n', x_data.isnull().sum())
mean_cols = x_data.mean()
x_data[numeric_cols] = x_data[numeric_cols].fillna(mean_cols)
print('填补后缺失值总数：{}'.format(x_data.isnull().sum().sum()))

# 分离数据
# iloc是Pandas的索引方法，允许通过位置索引来选择行和列。
x_train = x_data.iloc[:x_train.shape[0], :]
x_test = x_data.iloc[x_train.shape[0]:, :]

# 标签进行log1p
plt.hist(y_train, bins=30)
plt.show()

y_train_log1p = np.log1p(y_train)
plt.hist(y_train_log1p, bins=30)
plt.show()

# 创建模型并训练
reg = RandomForestRegressor()
result = reg.fit(x_train, y_train_log1p)
y_train_log1p_pred = reg.predict(x_train)
y_test_log1p_pred = reg.predict(x_test)
y_train_pred = np.expm1(y_train_log1p_pred)
y_test_pred = np.expm1(y_test_log1p_pred)
R2 = reg.score(x_train, y_train_log1p) # R²反映了模型解释数据方差的比例，值的范围是0到 1，值越接近1表示模型的拟合效果越好。
Cro_val_score = cross_val_score(reg, x_train, y_train_log1p, cv=10).mean()
MSE = mean_squared_error(y_train_log1p, y_train_log1p_pred)

print('R2:{}'.format(R2))
print('Cross validation score:{}'.format(Cro_val_score))
print(f'MSE:{MSE}')

# 可视化
plt.plot(y_train, label='y_true', linestyle='', marker='o', markerfacecolor='none')
plt.plot(y_train_pred, label='y_pred', linestyle='', marker='*')
plt.legend(loc='best')
plt.title('Train results(R2:{:.4f}, Cross val score:{:.4f}, MSE:{:.4f})'.format(R2, Cro_val_score, MSE))
plt.show()

plt.plot(y_test_pred)
plt.title('Test predictions')
plt.show()

# 提交结果
submission = pd.DataFrame({'Id':test_id, 'SalePrice':np.round(y_test_pred).astype(int)})
submission.to_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\House Prices - Advanced Regression Techniques\Submission.csv', index=False)
