import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def dist_eucl(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def get_closest_dist(point, centroid):
    min_dist = np.inf
    for j in range(len(centroid)):
        distance = dist_eucl(point, centroid[j])
        if distance < min_dist:
            min_dist = distance
    return min_dist

def kpp_cent(data_mat, k):
    data_set = np.asarray(data_mat)  # 将矩阵转换为数组
    centroid = [data_set[random.randint(0, len(data_set)-1)]]
    d = [0] * len(data_set)
    for _ in range(1, k):
        total = 0.0
        for i in range(len(data_set)):
            d[i] = get_closest_dist(data_set[i], centroid)
            total += d[i]
        total *= random.random()
        for j in range(len(d)):
            total -= d[j]
            if total > 0:
                continue
            centroid.append(data_set[j])
            break
    return np.mat(centroid)

def kpp_means(data_mat, k, dist=dist_eucl, create_cent=kpp_cent):
    m = np.shape(data_mat)[0]
    cluste_assment = np.mat(np.zeros((m, 2)))
    centroid = create_cent(data_mat, k)  # 直接调用传递的函数
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_index = -1
            min_dist = np.inf
            for j in range(k):
                distance = dist(data_mat[i, :], centroid[j, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            if cluste_assment[i, 0] != min_index:
                cluster_changed = True
                cluste_assment[i, :] = min_index, min_dist ** 2
        for j in range(k):
            per_data_set = data_mat[np.nonzero(cluste_assment[:, 0].A == j)[0]]
            centroid[j, :] = np.mean(per_data_set, axis=0)
    return centroid, cluste_assment

# 聚类的类数在: centroid, cluste_assment = kpp_means(np.mat(data_mat), 3)，(,n)这里 n = 3
# 哪两列设置为x轴和y轴，在：
# plt.plot(data_mat[:, 1], data_mat[:, 6], 'o', color='#C06C84', markeredgecolor='k', markersize=8)
# plt.text(point[1] + 5, point[6] - 5, labels[i], fontsize=8, ha='left', va='top', fontproperties='Simsun')
# plt.plot(per_data_set[:, 1], per_data_set[:, 6], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=8)
# plt.plot(centroid[:, 1], centroid[:, 6], '*', color='#F4A261', markersize=12)
# plt.text(point[1] + 5, point[6] - 5, labels[i], fontsize=8, ha='left', va='top', fontproperties='Simsun')
#这五个地方要同步设置([,1],[,6])、([,1],[,6])、([,1],[,6])、([,1],[,6])、([,1],[,6])8列的话可以从0到7里面选两个

# Load data
file_path = "1.csv"
df = pd.read_csv(file_path, encoding='GBK')

# Convert DataFrame to numpy array
data_mat = df.iloc[:, 1:].values
labels = df.iloc[:, 0].values  # 获取第一列标签

# Perform clustering
centroid, cluste_assment = kpp_means(np.mat(data_mat), 2)

# Plot clustering results for the first dataset
plt.figure(figsize=(8, 8), dpi=120)
plt.plot(data_mat[:, 1], data_mat[:, 6], 'o', color='#C06C84', markeredgecolor='k', markersize=8)
plt.title("原始数据", fontsize=15, fontproperties='Simsun')
plt.xlabel("xxx", fontsize=12)  # Add xlabel
plt.ylabel("yyy", fontsize=12)  # Add ylabel
plt.tight_layout()

# 在每个数据点旁边添加标签
for i, point in enumerate(data_mat):
    plt.text(point[1] + 5, point[6] - 5, labels[i], fontsize=8, ha='left', va='top', fontproperties='Simsun')

plt.savefig('原始数据.svg', format='svg', bbox_inches='tight')
plt.show()

# Plot clustering results
plt.figure(figsize=(8, 8), dpi=120)
k = np.shape(centroid)[0]
colors = plt.cm.Spectral(np.linspace(0, 1, k))  # 使用颜色映射生成k个颜色

# Plot points with different colors for each cluster
for i in range(k):
    per_data_set = data_mat[np.nonzero(cluste_assment[:, 0].A == i)[0]]
    plt.plot(per_data_set[:, 1], per_data_set[:, 6], 'o', markerfacecolor=tuple(colors[i]),
             markeredgecolor='k', markersize=8)

# Plot centroids with the same color as the clusters
for i in range(k):
    plt.plot(centroid[i, 1], centroid[i, 6], '*', markersize=12, color=colors[i])

plt.title("k-Means++ Cluster, k = {}".format(k), fontsize=15)
plt.xlabel("xxx", fontsize=12)  # Add xlabel
plt.ylabel("yyy", fontsize=12)  # Add ylabel
plt.tight_layout()

# 在每个数据点旁边添加标签
for i, point in enumerate(data_mat):
    plt.text(point[1] + 5, point[6] - 5, labels[i], fontsize=8, ha='left', va='top', fontproperties='Simsun')

plt.savefig('k-Means++ Cluster.svg', format='svg', bbox_inches='tight')
plt.show()

# Add cluster labels to the DataFrame
df['Cluster'] = cluste_assment[:, 0].A.flatten()

# Save DataFrame to CSV
result_file_path = "clustering_results.csv"
df.to_csv(result_file_path, index=False, encoding='GBK')
