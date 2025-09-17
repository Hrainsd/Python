import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Cluster\K-means\1.csv"
df = pd.read_csv(file_path, encoding='GBK')

# Convert DataFrame to numpy array
data_mat = df.iloc[:, 1:].values

# Calculate inertia for different values of K
inertia_values = []
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_mat)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow method curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia_values, marker='o', linestyle='-')
plt.title('Elbow Method Curve')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(False)
plt.savefig('Elbow Method Curve.svg', format='svg', bbox_inches='tight')
plt.show()

# Apply K-means algorithm
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters)  # You can specify the number of clusters you want
kmeans.fit(data_mat)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#同步更改以下三个地方的列号
#plt.scatter(data_mat[:, 0], data_mat[:, 5], alpha=0.5)
#plt.scatter(data_mat[labels == i, 0], data_mat[labels == i, 5], label=f'Cluster {i + 1}', alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 5], marker='*', s=200, color='#B2B2FF', label='Centroids')

# Generate colors for each cluster
colors = plt.cm.get_cmap('tab10', n_clusters)

# Plot the data before clustering
plt.figure(figsize=(8, 6))
plt.scatter(data_mat[:, 1], data_mat[:, 6], alpha=0.5)
plt.title('Scatter Plot of Data before Clustering')
plt.xlabel('Feature 2')
plt.ylabel('Feature 7')
plt.savefig('Before k-Means Cluster.svg', format='svg', bbox_inches='tight')
plt.show()

# Plot the data after clustering
plt.figure(figsize=(8, 6))

# Plot points with different colors for each cluster
for i in range(n_clusters):
    plt.scatter(data_mat[labels == i, 1], data_mat[labels == i, 6],
                color=colors(i), label=f'Cluster {i+1}', alpha=0.5)

# Plot centroids with the same color as the clusters
plt.scatter(centroids[:, 1], centroids[:, 6], marker='*', s=200,
            color=[colors(i) for i in range(n_clusters)], label='Centroids')

plt.title('Scatter Plot of Data after Clustering')
plt.xlabel('Feature 2')
plt.ylabel('Feature 7')
plt.legend()
plt.savefig('k-Means Cluster.svg', format='svg', bbox_inches='tight')
plt.show()

# Save DataFrame to CSV
result_file_path = r"clustering_results.csv"
df['Cluster_Labels'] = labels
df.to_csv(result_file_path, index=False, encoding='GBK')
