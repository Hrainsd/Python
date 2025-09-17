import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def rand_cent(data_mat, k):
    n = data_mat.shape[1]
    centroids = np.zeros((k, n))
    if not np.any(data_mat):
        return centroids
    for j in range(n):
        min_j = np.min(data_mat[:, j])
        range_j = float(np.max(data_mat[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k)
    return centroids

def dist_eucl(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def k_means(data_mat, k, dist=dist_eucl, create_cent=rand_cent):
    m, n = data_mat.shape
    cluster_assment = np.zeros((m, 2))  # Cluster index, distance
    centroids = create_cent(data_mat, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_index = -1
            min_dist = np.inf
            for j in range(k):
                distance = dist(data_mat[i, :], centroids[j, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
        for j in range(k):
            per_data_set = data_mat[np.nonzero(cluster_assment[:, 0] == j)[0]]
            centroids[j, :] = np.mean(per_data_set, axis=0)
    return centroids, cluster_assment

def bi_kmeans(data_mat, k, dist=dist_eucl):
    m = data_mat.shape[0]
    cluster_assment = np.zeros((m, 2))  # Cluster index, distance
    centroid0 = np.mean(data_mat, axis=0).tolist()
    cent_list = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist(np.mat(centroid0), data_mat[j, :]) ** 2
    while len(cent_list) < k:
        lowest_sse = np.inf
        for i, centroid in enumerate(cent_list):
            ptsin_cur_cluster = data_mat[np.nonzero(cluster_assment[:, 0] == i)[0], :]
            centroid_mat, split_cluster_ass = k_means(ptsin_cur_cluster, k=2)
            sse_split = np.sum(split_cluster_ass[:, 1])
            sse_nonsplit = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0] != i)[0], 1])
            if sse_split + sse_nonsplit < lowest_sse:
                best_cent_tosplit = i
                best_new_cents = centroid_mat
                best_cluster_ass = split_cluster_ass.copy()
                lowest_sse = sse_split + sse_nonsplit
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0] == 1)[0], 0] = len(cent_list)
        best_cluster_ass[np.nonzero(best_cluster_ass[:, 0] == 0)[0], 0] = best_cent_tosplit
        cent_list[best_cent_tosplit] = best_new_cents[0, :].tolist()
        cent_list.append(best_new_cents[1, :].tolist())
        cluster_assment[np.nonzero(cluster_assment[:, 0] == best_cent_tosplit)[0], :] = best_cluster_ass
    return np.array(cent_list), cluster_assment

# 聚类的类数在centroid, cluster_assment = bi_kmeans(data_mat, 3)，(,n)这里 n = 3
# 哪两列设置为x轴和y轴，在：
# plt.plot(data_mat[:, 1], data_mat[:, 2], 'o', color='#C06C84', markeredgecolor='k', markersize=8)
# plt.plot(point[1], point[2], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8)
# plt.text(point[1] + 5, point[2] - 5, label, fontsize=8, ha='left', va='top', fontproperties='Simsun')
# plt.plot(centroid[:, 1], centroid[:, 2], '*', color = '#F4A261', markersize=12)
#这四个地方要同步设置([,1],[,2])、([1],[2])、([1],[2])、([,1],[,2])，8列的话可以从0到7里面选两个

# Load data from CSV file
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Cluster\Bi-kmeans\1.csv"
df = pd.read_csv(file_path, encoding='gbk')

# Convert DataFrame to numpy array
data_mat = df.iloc[:, 1:].values
labels = df.iloc[:, 0].values  # 获取第一列标签

# Perform bi-kMeans clustering
centroid, cluster_assment = bi_kmeans(data_mat, 3)
print(centroid)

# Plot clustering results for the first dataset
plt.figure(figsize=(8, 8), dpi=120)
plt.plot(data_mat[:, 0], data_mat[:, 1], 'o', color='#C06C84', markeredgecolor='k', markersize=8)
plt.title("原始数据", fontsize=15, fontproperties='Simsun')
plt.xlabel("Carlos Alcaraz", fontsize=12)  # Add xlabel
plt.ylabel("Novak Djokovic", fontsize=12)  # Add ylabel
plt.tight_layout()

# 在每个数据点旁边添加标签
for i, point in enumerate(data_mat):
    plt.text(point[0] + 5, point[1] - 5, labels[i], fontsize=12, ha='left', va='top', fontproperties='Simsun')

plt.savefig('原始数据.svg', format='svg', bbox_inches='tight')
plt.show()

# Plot clustering results
plt.figure(figsize=(8, 8), dpi=120)
k = centroid.shape[0]
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, k)]

# 绘制每个簇的点
for i, col in zip(range(k), colors):
    per_data_set = data_mat[np.nonzero(cluster_assment[:, 0] == i)[0]]
    labels = df.iloc[np.nonzero(cluster_assment[:, 0] == i)[0], 0]  # Get labels from the first column
    for label, point in zip(labels, per_data_set):
        plt.plot(point[0], point[1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8)
        plt.text(point[0] + 5, point[1] - 5, label, fontsize=12, ha='left', va='top', fontproperties='Simsun')

# 绘制质心
for i in range(k):
    plt.plot(centroid[i, 0], centroid[i, 1], '*', color=colors[i], markersize=12)  # 使用对应的颜色

plt.title("Bi-KMeans Cluster, k = {}".format(k), fontsize=15)
plt.xlabel("Carlos Alcaraz", fontsize=12)  # Add xlabel
plt.ylabel("Novak Djokovic", fontsize=12)  # Add ylabel
plt.tight_layout()
plt.savefig('Bi-Kmeans Cluster.svg', format='svg', bbox_inches='tight')
plt.show()

# Add cluster labels to the DataFrame
df['Cluster'] = cluster_assment[:, 0]

# Save DataFrame to CSV
result_file_path = "Bi_Kmeans_results.csv"
df.to_csv(result_file_path, index=False, encoding='GBK')
