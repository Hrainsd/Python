# spearman相关系数不受很多条件的限制，哪里都能用
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, shapiro
from matplotlib.colors import LinearSegmentedColormap

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Spearman_Correlation_Coefficient\1.csv"
df = pd.read_csv(file_path)

# 计算Spearman相关系数和p值
spearman_corr, p_value = spearmanr(df)
print("Spearman correlation coefficient matrix:")
print(spearman_corr)
print("p-values matrix:")
print(p_value)

# 将Spearman相关系数保存到CSV文件
spearman_corr_df = pd.DataFrame(spearman_corr, index=df.columns, columns=df.columns)
spearman_corr_df.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Spearman_Correlation_Coefficient\相关系数.csv")

# 将p值保存到CSV文件
p_value_df = pd.DataFrame(p_value, index=df.columns, columns=df.columns)
p_value_df.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Spearman_Correlation_Coefficient\p值.csv")

# 判断数据是否服从正态分布
shapiro_matrix = pd.DataFrame(index=df.columns, columns=['Shapiro-Wilk statistic', 'p-value', 'Normality'])

for col in df.columns:
    statistic, shapiro_p_value = shapiro(df[col])
    shapiro_matrix.loc[col, 'Shapiro-Wilk statistic'] = statistic
    shapiro_matrix.loc[col, 'p-value'] = shapiro_p_value
    if shapiro_p_value > 0.05:
        shapiro_matrix.loc[col, 'Normality'] = 'Yes'
    else:
        shapiro_matrix.loc[col, 'Normality'] = 'No'

print("\nShapiro-Wilk test results:")
print(shapiro_matrix)

# 创建渐变的颜色映射
colors = ["#7FFFD4", "#B9EBD3", "#99CCFF", "#0099CC"]
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

# 绘制热图
plt.figure(figsize=(16, 8))
sns.heatmap(spearman_corr_df, annot=False, cmap=cmap, fmt=".4f")

# 添加星号表示p值
for i in range(len(spearman_corr_df)):
    for j in range(len(spearman_corr_df)):
        p_value = p_value_df.iloc[i, j]
        if p_value < 0.01:
            symbol = "***"
        elif p_value < 0.05:
            symbol = "**"
        elif p_value < 0.1:
            symbol = "*"
        else:
            symbol = ""
        plt.text(j+0.5, i+0.8, symbol, ha='center', va='center', color='black', fontsize=10)

plt.title('Correlation Heatmap with significance')
plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Spearman_Correlation_Coefficient\显著性星号热图.svg")
plt.show()
plt.close()

# 绘制矩阵散点图
scatter_matrix = pd.plotting.scatter_matrix(df, figsize=(10, 10), color='#F7A8A8', hist_kwds={'color': '#FED7D7'})  # 设置散点颜色，直方图颜色，对角线为核密度估计图

# 获取对角线右上部分的索引
upper_indices = np.triu_indices(len(df.columns), k=1)

for i, j in zip(*upper_indices):
    scatter_matrix[i, j].set_facecolor('#F7A8A8')  # 设置背景颜色为透明

# 添加相关系数的值到对角线上方的各个正方形里
for i, col in enumerate(df.columns):
    for j, row in enumerate(df.columns):
        if j > i:
            ax = scatter_matrix[i][j]
            ax.annotate(f'{spearman_corr_df.loc[row, col]:.3f}', (0.5, 0.5), color='black', ha='center', va='center', xycoords='axes fraction')

# 隐藏坐标轴刻度值
for ax in scatter_matrix.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

# 调整标题位置
plt.suptitle('Scatterplot Matrix', y=0.91)
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Spearman_Correlation_Coefficient\矩阵散点图.svg")
plt.show()
plt.close()
