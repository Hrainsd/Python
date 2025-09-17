# 最大值、最小值、平均值和总和
import numpy as np
import pandas as pd

data = np.array([[1 ,5, 7],
                 [2, 6 ,8],
                 [5, 2, 1]])
data = pd.DataFrame(data)

# 每行的话，指定axis=1
max_per_column = data.max()
min_per_column = data.min()
mean_per_column = data.mean()
sum_per_column = data.sum()

# 打印结果
print("每列的最大值：\n", max_per_column)
print("每列的最小值：\n", ",".join(map(str, min_per_column)))
print("每列的平均值：\n", ",".join(map(str, mean_per_column)))
print("每列的总和：\n", ",".join(map(str, sum_per_column)))


# 归一化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
fit_data = scaler.fit(data) # 生成min和max
output1 = scaler.transform(data) # 通过接口导出结果
print(output1)

# 特征数量巨大，使用partical_fit
fit_data1 = scaler.partial_fit(data)
output2 = scaler.transform(data)
print(output2)

# fit_transform将fit和transform两步化为一步
output3 = scaler.fit_transform(data)
conv_to_input = scaler.inverse_transform(output3)
print(output3)
print(conv_to_input)


# 标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
fit_data = scaler.fit(data)
output1 = scaler.transform(data)
fit_data1 = scaler.partial_fit(data)
output2= scaler.transform(data)
output3 = scaler.fit_transform(data)
conv_to_input = scaler.inverse_transform(output3)
print(output1)
print(output2)
print(output3)
print(conv_to_input)
print(scaler.mean_)
print(scaler.var_)
print(output3.mean())
print(output3.var())


# 缺失值处理
from sklearn.impute import SimpleImputer

x = [[1, 4, np.nan, 8],
     [np.nan, 1, 5, 3],
     [2, np.nan, 8, 8],
     [6, 2, 1, np.nan]]

# 第0列数据
x_0 = [row[0] for row in x]
x_0 = np.array(x_0).reshape(-1, 1)

imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer2 = SimpleImputer(strategy='median')
imputer3 = SimpleImputer(strategy='constant', fill_value=0)
imputer4 = SimpleImputer(strategy='most_frequent')

result1 = imputer1.fit_transform(x)
result2 = imputer2.fit_transform(x)
result3 = imputer3.fit_transform(x)
result4 = imputer4.fit_transform(x_0)
print(result1)
print(result2)
print(result3)
print(result4)


# 编码与哑变量
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer

# LabelEncoder---用于标签数据
x = np.array([['a', 'big', np.nan, 'yes'],
     ['b', 'small', 5, 'no'],
     ['b', 'small', 8, 'unknow'],
     ['a', 'small', 1, 'yes']])
encoder = LabelEncoder()
result = encoder.fit_transform(x[:, -1])
conv_to_input = encoder.inverse_transform(result)
x[:, -1] = result
print(result)
print(conv_to_input)
print(x)

# OrdinalEncoder---用于特征数据
x = np.array([['a', 'big', np.nan, 'yes'],
     ['b', 'small', 5, 'no'],
     ['b', 'small', 8, 'unknow'],
     ['a', 'small', 1, 'yes']])
encoder = OrdinalEncoder()
result = encoder.fit_transform(x[:, [0, 1]])
conv_to_input = encoder.inverse_transform(result)
x[:, 0 : 2] = result
print(result)
print(conv_to_input)
print(x)

# OneHotEncoder---one-hot编码，用于特征和标签数据，都可以
x = np.array([['a', 'big', np.nan, 'yes'],
     ['b', 'small', 5, 'no'],
     ['b', 'small', 8, 'unknow'],
     ['a', 'small', 1, 'yes']])
encoder = OneHotEncoder(categories='auto')
result = encoder.fit_transform(x[:, 0 : 2]).toarray()
conv_to_input = encoder.inverse_transform(result)
x = np.append(x, x[:, 2 : 4], axis=1)
x[:, 0 : 4] = result
print(result)
print(conv_to_input)
print(x)
print(encoder.get_feature_names_out())

x = pd.DataFrame(x, columns=['f_1_a', 'f_1_b', 'f_2_big', 'f_2_small', 'f_3', 'label'])
x.drop('f_3', axis=1, inplace=True)
x.columns = ['Feature_a', 'Feature_b', 'Feature_big', 'Feature_small', 'Label']
print(x)

# 二值化与分段/分箱
x = np.array([['a', 'big', 4, 'yes'],
     ['b', 'small', 5, 'no'],
     ['b', 'small', 8, 'unknow'],
     ['a', 'small', 1, 'yes']])
encoder = Binarizer(threshold=5)
input = x[:, 2].astype(int).reshape(-1, 1)
result = encoder.fit_transform(input)
result = result.reshape(-1)
x[:, 2] = result
print(result)
print(x)

x = np.array([['a', 'big', 4, 'yes'],
     ['b', 'small', 5, 'no'],
     ['b', 'small', 8, 'unknow'],
     ['a', 'small', 1, 'yes']])
encoder = KBinsDiscretizer(n_bins=2, encode='onehot', strategy='quantile')
input = x[:, 2].astype(int).reshape(-1, 1)
result = encoder.fit_transform(input).toarray()
conv_to_input = encoder.inverse_transform(result)
x = np.insert(x, 4, x[:, 3], axis=1)
x[:, 2 : 4] = result
print(result)
print(conv_to_input)
print(x)
