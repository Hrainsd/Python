# 连续数据，正态分布，线性关系，用pearson相关系数是最恰当的
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro, pearsonr

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Pearson_Correlation_Coefficient\1.csv"
df = pd.read_csv(file_path)

# Selecting relevant columns
# df = df.iloc[:, 3:-1]

# 执行JB检验(n>30)
jb_test_statistic, jb_p_value = jarque_bera(df)
print("JB检验统计量:", jb_test_statistic)
print("JB检验p值:", jb_p_value)

# 执行Shapiro-Wilk检验(3≤n≤50)
shapiro_test_statistic, shapiro_p_value = shapiro(df)
print("Shapiro-Wilk检验统计量:", shapiro_test_statistic)
print("Shapiro-Wilk检验p值:", shapiro_p_value)

if jb_p_value >= 0.05:
    print("数据服从正态分布 (JB检验)")
else:
    print("数据不服从正态分布 (JB检验)")

if shapiro_p_value >= 0.05:
    print("数据服从正态分布 (Shapiro-Wilk检验)")
else:
    print("数据不服从正态分布 (Shapiro-Wilk检验)")

# 设置subplot布局
num_cols = df.shape[1]
num_rows = (num_cols + 1) // 2  # 向上取整，以适应不同列数

# 创建画布和子图
fig, axes = plt.subplots(2, num_rows, figsize=(22, num_rows * 1))
plt.subplots_adjust(hspace=0.2, wspace=1.2)  # 调整上下图和左右图之间的距离
axes = axes.flatten()

# 循环绘制QQ图
for i in range(num_cols):
    col = df.columns[i]
    ax = axes[i]
    stats.probplot(df[col], dist="norm", plot=ax)  # 绘制QQ图并设置线条颜色
    x, y = ax.get_lines()[0].get_data()
    ax.plot(x, y, marker='o', markersize=5, color='#FFA07A', markerfacecolor='#FFA07A', linestyle='', label='Data points')
    ax.get_lines()[1].set_color('#66C7B4')
    ax.set_title(f'QQ Plot - {col}')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Ordered Values')
    ax.grid(False)

# 隐藏多余的子图
for i in range(num_cols, len(axes)):
    fig.delaxes(axes[i])

plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Pearson_Correlation_Coefficient\QQ图.svg")
plt.show()
plt.close()

# 计算皮尔逊相关系数和p值
corr_matrix = df.corr(method='pearson')
print("皮尔逊相关系数：\n", corr_matrix)

# 将皮尔逊相关系数保存到CSV文件
corr_matrix.to_csv('2.csv')

# 计算p值
p_values = pd.DataFrame(index=df.columns, columns=df.columns)
for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            _, p_value = pearsonr(df[col1], df[col2])
            p_values.loc[col1, col2] = p_value

# 输出p值矩阵
print("\np值矩阵：\n", p_values)

# 将p值保存到CSV文件
corr_matrix.to_csv('3.csv')

# 创建渐变的颜色映射
colors = ["#7FFFD4", "#B9EBD3", "#99CCFF", "#0099CC"]
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

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
            ax.annotate(f'{corr_matrix.loc[row, col]:.3f}', (0.5, 0.5), color='black', ha='center', va='center', xycoords='axes fraction')

# 隐藏坐标轴刻度值
for ax in scatter_matrix.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

# 调整标题位置
plt.suptitle('Scatter Matrix', y=0.91)
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Pearson_Correlation_Coefficient\矩阵散点图.svg")
plt.show()
plt.close()

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".4f")

# 添加星号表示p值
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        p_value = p_values.iloc[i, j]
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
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Pearson_Correlation_Coefficient\显著性星号热图.svg")
plt.show()
plt.close()
