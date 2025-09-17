# PCA, SVD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 创建数据集
iris = load_iris()
x = iris.data
y = iris.target
y_names = iris.target_names

pca = PCA(n_components=2)
result = pca.fit_transform(x)
print(result)
print(pca.explained_variance_) # 可解释性方差，越大越重要
print(pca.explained_variance_ratio_) # 可解释性方差占原始数据总信息量的比值
print(pca.explained_variance_ratio_.sum()) # 可解释性方差占比之和


colors = ['r', 'g', 'b']
for i, color, y_name in zip([0, 1, 2], colors, y_names):
    plt.scatter(result[y == i, 0], result[y == i, 1], color=color, label=y_name)
plt.legend()
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 获取最佳维度
pca1 = PCA() # 不填n_components时，默认为特征数量和样本数量的最小值，一般为特征数量，找到需要降到的维度
result1 = pca1.fit_transform(x)
result1_cumsum = np.cumsum(pca1.explained_variance_ratio_)
print(result1_cumsum)
plt.plot(range(1, 5), result1_cumsum)
plt.title('Cumulus Variance Ratio curve')
plt.show()
