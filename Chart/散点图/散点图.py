# 实心散点图
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, scale_color_manual, scale_fill_manual, scale_x_continuous, scale_y_continuous, guides, theme,stat_summary,element_blank,element_text,element_line

# Generate random data
np.random.seed(42)
n_samples = 100
dt = pd.DataFrame({
    'PC1': np.random.normal(0, 2, n_samples),
    'PC2': np.random.normal(0, 2, n_samples),
    'Diagnosis': np.random.choice(['A', 'B'], n_samples)
})

# Create the plot
p1 = (
    ggplot(dt, aes(x='PC1', y='PC2', fill='Diagnosis')) +
    stat_summary(fun_y=np.mean, geom='point', size=3, color='white', shape=1, fill='white') +
    geom_point(size=3, alpha=0.7) +
    scale_color_manual(name="Category", values=["#FF9999", "#c77cff"]) +
    scale_fill_manual(name="Category", values=["#FF9999", "#c77cff"]) +
    scale_x_continuous(expand=(0.7, 0.7), limits=(-10, 5)) +
    scale_y_continuous(expand=(0.5, 0.5), limits=(-7.5, 5)) +
    guides(x='axis_truncated', y='axis_truncated')+
    theme(panel_background=element_blank(),  # Set background color to white
          legend_position=(0.5, 1),  # Move legend to the top and center
          legend_direction='horizontal',  # Display legend horizontally
          legend_box='horizontal',  # Display legend horizontally
          axis_title=element_text(size=10),  # Set axis title text size
          axis_text=element_text(size=8),  # Set axis tick label text size
          legend_text=element_text(size=8),  # Set legend text size
          panel_grid_major=element_blank(),  # Remove major grid lines
          panel_grid_minor=element_blank(),  # Remove minor grid lines
          axis_line=element_line(color="black"),  # Set axis line color
          legend_key=element_blank(),  # Remove legend keys
          axis_line_x=element_line(color="black"),  # Set x-axis line color
          axis_line_y=element_line(color="black")  # Set y-axis line color
          )
)

# Show the plot
print(p1)



# 透明散点图
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, stat_summary, scale_color_manual, scale_fill_manual, scale_x_continuous, scale_y_continuous, guides, theme, element_blank, element_text, element_line

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart2_san_dian_tu\1.csv"
df = pd.read_csv(file_path)
df['category'] = df['category'].map({'A': 'A', 'B': 'B'})

# Create the plot
p1 = (
    ggplot(df, aes(x='pc1', y='pc2', color='category', fill='category')) +
    stat_summary(fun_y=np.mean, geom='point', size=3, color='white', shape=1, fill='white') +
    geom_point(size=3, alpha=0.7) +
    scale_color_manual(name="Category", values=["#FF9999", "#c77cff"]) +
    scale_fill_manual(name="Category", values=["#FF9999", "#c77cff"]) +
    scale_x_continuous(expand=(0.7, 0.7), limits=(-10, 5)) +
    scale_y_continuous(expand=(0.5, 0.5), limits=(-7.5, 5)) +
    guides(x='axis_truncated', y='axis_truncated') +
    theme(
        panel_background=element_blank(),  # Set background color to white
        legend_position=(0.5, 1),  # Move legend to the top and center
        legend_direction='horizontal',  # Display legend horizontally
        legend_box='horizontal',  # Display legend horizontally
        axis_title=element_text(size=10),  # Set axis title text size
        axis_text=element_text(size=8),  # Set axis tick label text size
        legend_text=element_text(size=8),  # Set legend text size
        panel_grid_major=element_blank(),  # Remove major grid lines
        panel_grid_minor=element_blank(),  # Remove minor grid lines
        axis_line=element_line(color="black"),  # Set axis line color
        legend_key=element_blank(),  # Remove legend keys
        axis_line_x=element_line(color="black"),  # Set x-axis line color
        axis_line_y=element_line(color="black")  # Set y-axis line color
    )
)

# Show the plot
print(p1)



# 带质心的散点图
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart2_san_dian_tu\1.csv"
df = pd.read_csv(file_path)

# 确保 'category' 列的正确映射
df['category'] = df['category'].map({'A': 'A', 'B': 'B'})

# 进行PCA降维，确保只选择数值列进行PCA
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df.select_dtypes(include='number')), columns=['PC1', 'PC2'])
df_pca['category'] = df['category']

# 设置颜色映射
colors = {'A': "#FF9999", 'B':  "#c77cff"}

# 绘制散点图
fig, ax = plt.subplots()

# 存储每个类别的质心坐标
centroids = {}

for category, color in colors.items():
    subset = df_pca[df_pca['category'] == category]
    centroid = subset[['PC1', 'PC2']].mean()  # 只计算数值列的均值
    centroids[category] = centroid

    # 调整数据点的位置
    ax.scatter(subset['PC1'] - 0.2, subset['PC2'] - 0.2, label=category, c=color, alpha=0.7, s=30)

    # 绘制类别内每个点到质心的连线
    for _, point in subset.iterrows():
        ax.plot([point['PC1'] - 0.2, centroid['PC1']], [point['PC2'] - 0.2, centroid['PC2']], linestyle='-', color=color, linewidth=0.3)

# 添加质心线
for category, color in colors.items():
    centroid = centroids[category]
    ax.plot([centroid['PC1']], [centroid['PC2']], marker='o', markersize=10, markerfacecolor=color, markeredgecolor='white', label=f'Centroid {category}', markeredgewidth=1.5)

# 设置坐标轴范围和刻度
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks([-5, 0, 5])
ax.set_yticks([-5, -2.5, 0, 2.5, 5])

# 移除右边和上边的坐标轴
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 将坐标轴断开
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))

# 调整布局，将数据点置于图形的中间
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# 设置图例位置，并移除框线
legend = ax.legend(loc='upper right', frameon=False)

# 设置图表标题
ax.set_title('PCA Plot with Centroid Lines')

# 保存图表为SVG文件
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 调整布局，使图形自动放在图片的正中央
plt.tight_layout()

# 显示图形
plt.show()



# 局部放大的散点图
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart2_san_dian_tu\1.csv"
df = pd.read_csv(file_path)
df['category'] = df['category'].map({'A': 'A', 'B': 'B'})

# 进行PCA降维
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df.iloc[:, :-1]), columns=['PC1', 'PC2'])
df_pca['category'] = df['category']

# 设置颜色映射
colors = {'A': "#FF9999", 'B':  "#c77cff"}

# 绘制散点图
fig, ax = plt.subplots()

# 存储每个类别的质心坐标
centroids = {}

for category, color in colors.items():
    subset = df_pca[df_pca['category'] == category]
    # 仅对数值型列计算均值
    centroid = subset[['PC1', 'PC2']].mean()
    centroids[category] = centroid

    # 调整数据点的位置
    ax.scatter(subset['PC1'] - 0.2, subset['PC2'] - 0.2, label=category, c=color, alpha=0.7, s=30)

    # 绘制类别内每个点到质心的连线
    for _, point in subset.iterrows():
        ax.plot([point['PC1'] - 0.2, centroid['PC1']], [point['PC2'] - 0.2, centroid['PC2']], linestyle='-', color=color, linewidth=0.3)

# 添加质心线
for category, color in colors.items():
    centroid = centroids[category]
    ax.plot([centroid['PC1']], [centroid['PC2']], marker='o', markersize=10, markerfacecolor=color, markeredgecolor='white', label=f'Centroid {category}', markeredgewidth=1.5)

# 设置坐标轴范围和刻度
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks([-5, 0, 5])
ax.set_yticks([-5, -2.5, 0, 2.5, 5])

# 移除右边和上边的坐标轴
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 将坐标轴断开
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))

# 调整布局，将数据点置于图形的中间
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# 设置图例位置，并移除框线
legend = ax.legend(loc='upper right', frameon=False)

# 设置图表标题
ax.set_title('PCA Plot with Centroid Lines')

# 创建放大区域
axins = inset_axes(ax, width="30%", height="30%", loc='lower left', bbox_to_anchor=(0.1, 0.7, 1, 1), bbox_transform=ax.transAxes)

# 使用所有数据点来创建散点图
for category, color in colors.items():
    subset = df_pca[df_pca['category'] == category]
    axins.scatter(subset['PC1'], subset['PC2'], c=color, alpha=0.7, s=30)

# 设置放大区域坐标轴范围和刻度
axins.set_xlim(1, 3)
axins.set_ylim(1, 3)
axins.set_xticks([1, 2, 3])
axins.set_yticks([1, 2, 3])

# 添加原图形被围起来的区域
rect = Rectangle((axins.get_xlim()[0], axins.get_ylim()[0]), axins.get_xlim()[1] - axins.get_xlim()[0], axins.get_ylim()[1] - axins.get_ylim()[0], linewidth=1, edgecolor='#48C0AA', facecolor='none')
ax.add_patch(rect)

# 连接底部左侧角
con1 = ConnectionPatch(xyA=(axins.get_xlim()[0], axins.get_ylim()[0]),
                       xyB=(rect.get_x(), rect.get_y()),
                       coordsA="data", coordsB="data",
                       axesA=axins, axesB=ax,
                       color='#48C0AA', linestyle='--', linewidth=1)
ax.add_patch(con1)

# 连接右上角
con2 = ConnectionPatch(xyA=(axins.get_xlim()[1], axins.get_ylim()[1]),
                       xyB=(rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()),
                       coordsA="data", coordsB="data",
                       axesA=axins, axesB=ax,
                       color='#48C0AA', linestyle='--', linewidth=1)
ax.add_patch(con2)

# 设置图表标题
ax.set_title('PCA Plot with Centroid Lines')

# 保存图表为SVG文件
plt.savefig('output_plot_2.svg', format='svg', bbox_inches='tight')

# 调整布局，使图形自动放在图片的正中央
plt.tight_layout()

# 显示图形
plt.show()
