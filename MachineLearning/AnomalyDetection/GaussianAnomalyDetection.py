import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 导入数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\AnomalyDetection\1.csv"
df = pd.read_csv(file_path, header=None)  # 添加header=None参数以表示数据没有列名

# 为DataFrame添加列名
num_cols = df.shape[1]
columns = [f"Feature_{i+1}" for i in range(num_cols)]
df.columns = columns

print(df)

# 对数据进行对数转换
df_log = np.log(df)

# 计算转换后数据的均值和协方差矩阵
mean = np.mean(df_log, axis=0)
covariance_matrix = np.cov(df_log, rowvar=False)

# 使用多元高斯分布拟合数据
multivariate_gaussian = multivariate_normal(mean=mean, cov=covariance_matrix)

# 计算每个样本点的概率密度
probabilities = multivariate_gaussian.pdf(df_log)

# 定义阈值，选择异常值
threshold = 0.01  # 可根据实际情况调整

# 找出概率密度小于阈值的样本点，认为是异常值
anomalies = df[probabilities < threshold]

# 输出异常值
print("异常值：")
print(anomalies)

# 画直方图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# 对数转换前的直方图
axes[0].hist(df.values.flatten(), bins=50, color='blue', alpha=0.7)
axes[0].set_title('Histogram of Original Data')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Frequency')

# 对数转换后的直方图
axes[1].hist(df_log.values.flatten(), bins=50, color='green', alpha=0.7)
axes[1].set_title('Histogram of Log-transformed Data')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
