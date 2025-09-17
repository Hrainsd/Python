# K-Means算法
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 创建数据集
x, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=42)

# facecolor用在plt.scatter(), markerfacecolor用在plt.plot()
plt.scatter(x[:, 0], x[:, 1], marker='o', color='black', facecolor='none')
plt.title('Scatter')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
y_pred_0 = kmeans.fit_predict(x) # fit_predict可以直接得到划分后的聚类簇
print(y_pred_0)

y_pred = kmeans.fit(x)
print(y_pred.labels_) # 所有样本被划分到的聚类簇
print(y_pred.cluster_centers_) # 所有聚类簇的质心数据
print(y_pred.inertia_) # 总距离平方和

y_pred_label = y_pred.labels_
centroid = y_pred.cluster_centers_
colors = ['r', 'g', 'b', 'c']
labels = ['1', '2', '3', '4']
figure, ax1 = plt.subplots(1) # 画布，子图对象
for i, color, label in zip(range(4), colors, labels):
    ax1.scatter(x[y_pred_label == i, 0], x[y_pred_label == i, 1], marker='o', color=color, s=30, label=label, alpha=0.2)
    ax1.scatter(centroid[i, 0], centroid[i, 1], marker='*', color=color, s=60)
plt.legend(loc='best')
plt.title('K-means Scatter')
plt.show()

# 轮廓系数（越接近1，聚出的类越合适，但也要根据特定的任务去看聚类是否合适）
from sklearn.metrics import silhouette_score, silhouette_samples

result1 = silhouette_score(x, y_pred_label) # 平均轮廓系数
result2 = silhouette_samples(x, y_pred_label) # 每个样本的轮廓系数
print(result1)
print(result2)
