# 主成分分析
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 导入数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\PCA\1.csv"
df = pd.read_csv(file_path)

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 执行主成分分析
pca = PCA()
pca.fit(scaled_data)

# 获取所需的主成分数量
n_components = pca.n_components_

# 将数据转换到新的特征空间
pca_data = pca.transform(scaled_data)

# 获取解释方差比率和主成分
explained_variance_ratio = pca.explained_variance_ratio_
principal_components = pca.components_

# 输出解释方差比率
print("解释方差比率:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"主成分 {i+1}: {ratio:.2f}")

# 累计解释方差贡献率
explained_variance_ratio_cumsum = explained_variance_ratio.cumsum()
print("累计解释方差贡献率:")
for i in range(len(explained_variance_ratio_cumsum)):
    print(f"主成分{i+1}的累计解释方差贡献率：{explained_variance_ratio_cumsum[i]}")

# 输出主成分
print("\n主成分:")
for i, component in enumerate(principal_components):
    print(f"主成分 {i+1}: {component}")

# 指定主成分的数量，将维度降低到n维
pca = PCA(n_components=0.90) # 保留90%的方差/累计贡献率0.9
pca_data_nd = pca.fit_transform(scaled_data)
print(pca_data_nd) # pca_data_nd 包含了降维后的数据
