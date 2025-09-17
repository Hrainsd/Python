import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Cov\1.csv"
df = pd.read_csv(file_path)

# 计算协方差矩阵
cov_matrix = df.cov()

# 自定义颜色渐变
custom_colors = ['#B9EBD3', '#D3F8E2', '#DAF7A6', '#E9C46A','#F4A261' ,'#E76F51']
cmap = LinearSegmentedColormap.from_list('custom', custom_colors, N=256)

# 使用seaborn绘制热图
plt.figure(figsize=(10, 8))

# 绘制热图并标注显著性
sns.heatmap(cov_matrix, annot=True, cmap=cmap, fmt='.2f', linewidths=0.5,
            annot_kws={"size": 10, "weight": "bold", "color": 'black'})

# 在每个单元格上添加显著性标签
for i in range(len(cov_matrix)):
    for j in range(len(cov_matrix.columns)):
        _, p_value = pearsonr(df[cov_matrix.index[i]], df[cov_matrix.columns[j]])
        if p_value < 0.01:
            plt.text(j + 0.5, i + 0.8, '**', ha='center', va='center', color='#0094C6', fontsize=10, fontweight='bold')
        elif p_value < 0.05:
            plt.text(j + 0.5, i + 0.8, '*', ha='center', va='center', color='#0094C6', fontsize=10, fontweight='bold')

plt.title('Covariance Matrix Heatmap')
plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Cov\1.svg")
plt.show()
plt.close()

# 保存协方差矩阵到1.2.csv（不包含列名）
file_path_1_2 = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Cov\1.2.csv"
cov_matrix.to_csv(file_path_1_2, index=False, header=False)

# 打印相应的p值矩阵
print("\nP-Values Matrix:")
p_values_matrix = pd.DataFrame(index=cov_matrix.index, columns=cov_matrix.columns)
for i in range(len(cov_matrix)):
    for j in range(len(cov_matrix.columns)):
        _, p_value = pearsonr(df[cov_matrix.index[i]], df[cov_matrix.columns[j]])
        p_values_matrix.iloc[i, j] = p_value

print(p_values_matrix)

# 保存相应的p值矩阵到1.3.csv（不包含列名）
file_path_1_3 = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Cov\1.3.csv"
p_values_matrix.to_csv(file_path_1_3, index=False, header=False)
