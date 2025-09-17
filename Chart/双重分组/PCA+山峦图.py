# 双重分组PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D  # Import Line2D for custom legend colors

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart17_shuang_chong_fen_zu_PCA+shan_luan_tu\1.csv"
df = pd.read_csv(file_path)

# 提取数据列
data = df[['pc1', 'pc2']]

# 标准化数据
data_standardized = (data - data.mean()) / data.std()

# 运行PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_standardized)

# 将PCA结果添加到DataFrame
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['group'] = df['group']
df_pca['cohort'] = df['cohort']

# 绘制PCA图
plt.figure(figsize=(10, 6))

# Modified scatter plot with color, shape, and size adjustments
for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df_pca[df_pca['group'] == group]

    for cohort, marker in zip(df['cohort'].unique(), ['o', '*']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        plt.scatter(x=cohort_data['PC1'], y=cohort_data['PC2'], c=color, marker=marker, s=100, label=None)

# 添加外围椭圆
for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df_pca[df_pca['group'] == group][['PC1', 'PC2']]

    # 计算椭圆的协方差矩阵
    cov_matrix = group_data.cov()

    # 计算椭圆的中心
    center = group_data.mean()

    # 创建椭圆
    ellipse = Ellipse(xy=center, width=2 * cov_matrix.iloc[0, 0] ** 0.5, height=2 * cov_matrix.iloc[1, 1] ** 0.5,
                      angle=0, edgecolor=color, fc='None', lw=2)

    # 添加椭圆到图中
    plt.gca().add_patch(ellipse)

# 添加图例
custom_legend_colors = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0999f', markersize=10),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#46bbc0', markersize=10)]

color_legend = plt.legend(custom_legend_colors, df['group'].unique(), title='Group', loc='upper right', frameon=False, bbox_to_anchor=(1.1, 1))

# 创建和添加Cohort图例
shape_legend = plt.legend(title='Cohort', labels=df['cohort'].unique(), loc='upper right', frameon=False, bbox_to_anchor=(1.1, 0.8))

# 设置'Cohort'图例中圆圈的颜色
for handle in shape_legend.legendHandles:
    handle.set_color('#92B4C8')

# 合并图例
plt.gca().add_artist(color_legend)
plt.gca().add_artist(shape_legend)

plt.title('Modified PCA Plot with Ellipses')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



# 双重分组山峦图
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Read CSV file
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart17_shuang_chong_fen_zu_PCA+shan_luan_tu\1.csv"
df = pd.read_csv(file_path)

# Extract data columns
data = df[['pc1', 'pc2']]

# Standardize data
data_standardized = (data - data.mean()) / data.std()

# Run PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_standardized)

# Add PCA results to DataFrame
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['group'] = df['group']
df_pca['cohort'] = df['cohort']

# Plot mountain plot
plt.figure(figsize=(10, 6))

for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df_pca[df_pca['group'] == group]

    for cohort, marker in zip(df['cohort'].unique(), ['^', 's']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        plt.plot(cohort_data['PC1'], cohort_data['PC2'], marker=marker, linestyle='-', color=color, markersize=8, label=None)

# Create legend for groups
group_legend = plt.legend(df['group'].unique(), title='Group', loc='upper right', frameon=False, bbox_to_anchor=(1.1, 1))

# Create and add legend for cohorts
cohort_legend = plt.legend(df['cohort'].unique(), title='Cohort', loc='upper right', frameon=False, bbox_to_anchor=(1.1, 0.8))

# Set colors for legend markers
for handle in cohort_legend.legendHandles:
    handle.set_color('#92B4C8')

# Add both legends to the plot
plt.gca().add_artist(group_legend)
plt.gca().add_artist(cohort_legend)

plt.title('Mountain Plot with Group and Cohort Differentiation')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



# 双重分组PCA+条形图
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.lines import Line2D  # Import Line2D for custom legend colors

# Read CSV file
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart17_shuang_chong_fen_zu_PCA+shan_luan_tu\1.csv"
df = pd.read_csv(file_path)

# Extract data columns
data = df[['pc1', 'pc2']]

# Standardize data
data_standardized = (data - data.mean()) / data.std()

# Run PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_standardized)

# Add PCA results to DataFrame
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['group'] = df['group']
df_pca['cohort'] = df['cohort']

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

# PC2 Density plot
for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df[df['group'] == group]
    for cohort, linetype in zip(df['cohort'].unique(), ['solid', 'dashed']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        axs[0, 0].hist(cohort_data['pc2'], bins=30, color=color, alpha=0.5, label=cohort, density=True, linestyle=linetype)

# 移除轴线和标签值
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].spines['bottom'].set_visible(False)
axs[0, 0].spines['left'].set_visible(False)

axs[0, 0].tick_params(axis='both', which='both', length=0)  # 移除刻度线
axs[0, 0].set_xticks([])  # 移除x轴刻度值
axs[0, 0].set_yticks([])  # 移除y轴刻度值
axs[0, 0].set_xlabel('')  # 移除x轴标签
axs[0, 0].set_ylabel('')  # 移除y轴标签

# Original PCA plot with ellipses
for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df_pca[df_pca['group'] == group]

    for cohort, marker in zip(df['cohort'].unique(), ['o', '*']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        axs[1, 0].scatter(x=cohort_data['PC1'], y=cohort_data['PC2'], c=color, marker=marker, s=100, label=None)

    group_data = df_pca[df_pca['group'] == group][['PC1', 'PC2']]
    cov_matrix = group_data.cov()
    center = group_data.mean()
    ellipse = Ellipse(xy=center, width=2 * cov_matrix.iloc[0, 0] ** 0.5, height=2 * cov_matrix.iloc[1, 1] ** 0.5,
                      angle=0, edgecolor=color, fc='None', lw=2)
    axs[1, 0].add_patch(ellipse)

# PC1 Density plot with 90-degree counterclockwise rotation
for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df[df['group'] == group]
    for cohort, linetype in zip(df['cohort'].unique(), ['solid', 'dashed']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        axs[1, 1].hist(cohort_data['pc1'], bins=30, color=color, alpha=0.5, label=cohort, density=True, linestyle=linetype, orientation='horizontal')  # 设置orientation参数为'horizontal'

# 移除轴线和标签值
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].spines['bottom'].set_visible(False)
axs[1, 1].spines['left'].set_visible(False)

axs[1, 1].tick_params(axis='both', which='both', length=0)  # 移除刻度线
axs[1, 1].set_xticks([])  # 移除x轴刻度值
axs[1, 1].set_yticks([])  # 移除y轴刻度值
axs[1, 1].set_xlabel('')  # 移除x轴标签
axs[1, 1].set_ylabel('')  # 移除y轴标签

# 添加图例
custom_legend_colors = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0999f', markersize=10),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#46bbc0', markersize=10)]

color_legend = plt.legend(custom_legend_colors, df['group'].unique(), title='Group', loc='upper right', frameon=False, bbox_to_anchor=(-0.1, 1))

# 创建和添加Cohort图例
custom_legend_colors_1 = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#92B4C8', markersize=10),
                        Line2D([0], [0], marker='*', color='w', markerfacecolor='#92B4C8', markersize=10)]

shape_legend = plt.legend(custom_legend_colors_1, df['cohort'].unique(), title='Cohort', loc='upper right', frameon=False, bbox_to_anchor=(-0.1, 0.8))

# 设置'Cohort'图例中圆圈的颜色
for handle in shape_legend.legendHandles:
    handle.set_color('#92B4C8')

# 合并图例
plt.gca().add_artist(color_legend)
plt.gca().add_artist(shape_legend)

# 移除轴线和标签值
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].spines['bottom'].set_visible(False)
axs[0, 1].spines['left'].set_visible(False)

axs[0, 1].tick_params(axis='both', which='both', length=0)  # 移除刻度线
axs[0, 1].set_xticks([])  # 移除x轴刻度值
axs[0, 1].set_yticks([])  # 移除y轴刻度值
axs[0, 1].set_xlabel('')  # 移除x轴标签
axs[0, 1].set_ylabel('')  # 移除y轴标签

plt.tight_layout()
plt.show()



# 双重分组PCA+山峦图
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

# Read CSV file
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart17_shuang_chong_fen_zu_PCA+shan_luan_tu\1.csv"
df = pd.read_csv(file_path)

# Extract data columns
data = df[['pc1', 'pc2']]

# Standardize data
data_standardized = (data - data.mean()) / data.std()

# Run PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_standardized)

# Add PCA results to DataFrame
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['group'] = df['group']
df_pca['cohort'] = df['cohort']

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

# PC2 Density plot
for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df[df['group'] == group]
    for cohort, linetype in zip(df['cohort'].unique(), ['solid', 'dashed']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        axs[0, 0].plot(cohort_data['pc2'], color=color, alpha=0.5, label=cohort, linestyle=linetype)
        axs[0, 0].fill_between(cohort_data.index, cohort_data['pc2'], color=color, alpha=0.1)

# Remove axes and labels
axs[0, 0].axis('off')

# Original PCA plot with ellipses
for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df_pca[df_pca['group'] == group]

    for cohort, marker in zip(df['cohort'].unique(), ['o', '*']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        axs[1, 0].scatter(x=cohort_data['PC1'], y=cohort_data['PC2'], c=color, marker=marker, s=100, label=None)

    group_data = df_pca[df_pca['group'] == group][['PC1', 'PC2']]
    cov_matrix = group_data.cov()
    center = group_data.mean()
    ellipse = Ellipse(xy=center, width=2 * cov_matrix.iloc[0, 0] ** 0.5, height=2 * cov_matrix.iloc[1, 1] ** 0.5,
                      angle=0, edgecolor=color, fc='None', lw=2)
    axs[1, 0].add_patch(ellipse)

axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)

for group, color in zip(df['group'].unique(), ['#f0999f', '#46bbc0']):
    group_data = df[df['group'] == group]
    for cohort, linetype in zip(df['cohort'].unique(), ['solid', 'dashed']):
        cohort_data = group_data[group_data['cohort'] == cohort]
        axs[1, 1].plot(cohort_data['pc1'], cohort_data.index, color=color, alpha=0.5, label=cohort, linestyle=linetype)
        axs[1, 1].fill_betweenx(cohort_data.index, cohort_data['pc1'], color=color, alpha=0.1)

# Remove axes and labels
axs[1, 1].axis('off')

# Legend
custom_legend_colors = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0999f', markersize=10),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#46bbc0', markersize=10)]

color_legend = plt.legend(custom_legend_colors, df['group'].unique(), title='Group', loc='upper right', frameon=False, bbox_to_anchor=(-0.1, 1))

custom_legend_colors_1 = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#92B4C8', markersize=10),
                        Line2D([0], [0], marker='*', color='w', markerfacecolor='#92B4C8', markersize=10)]

shape_legend = plt.legend(custom_legend_colors_1, df['cohort'].unique(), title='Cohort', loc='upper right', frameon=False, bbox_to_anchor=(-0.1, 0.9))

for handle in shape_legend.legendHandles:
    handle.set_color('#92B4C8')

plt.gca().add_artist(color_legend)
plt.gca().add_artist(shape_legend)

# Remove axes and labels
axs[0, 1].axis('off')

plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

plt.show()
