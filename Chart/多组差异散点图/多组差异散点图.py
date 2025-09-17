# 多组差异散点图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart6_duo_zu_cha_yi_san_dian_tu\1.csv"
df = pd.read_csv(file_path)

# 获取数据列 (//n, n为大组内的小组数)
group_count = len(df.columns) // 2
data_points_per_subgroup = len(df)

# 转换数据格式
data = []
labels = []

for i in range(group_count):
    group_data_A = df.iloc[:, i * 2].values
    group_data_B = df.iloc[:, i * 2 + 1].values

    data.append((group_data_A, group_data_B))
    labels.append(f'Group {i + 1}')

# 计算数据的最大值
max_value = max(np.max(np.concatenate([data[i][0] for i in range(group_count)])),
                np.max(np.concatenate([data[i][1] for i in range(group_count)])))

# 计算'a'的值，可以根据需要调整
a_value = max_value * 1.2

# 获取sci颜色映射
cmap = plt.get_cmap('coolwarm')
norm = Normalize(vmin=0, vmax=group_count-1)

# 绘制差异散点图
fig, (ax_upper, ax_lower) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'hspace': 0.05})

# 上半部分绘制A组数据
for i in range(group_count):
    ax_upper.scatter(np.full_like(data[i][0], i * 2), data[i][0], label=f'Group {i + 1}A', color=cmap(norm(i)), marker='o', linestyle='None')

# 下半部分绘制B组数据
for i in range(group_count):
    ax_lower.scatter(np.full_like(data[i][1], i * 2), data[i][1], label=f'Group {i + 1}B', color=cmap(norm(i)), marker='o', linestyle='None')

# 添加上下部分组内连线
for i in range(group_count):
    ax_upper.plot([i * 2, i * 2], [np.min(data[i][0]), np.max(data[i][0])], color=cmap(norm(i)), linestyle='-', linewidth=1, alpha=0.7)
    ax_lower.plot([i * 2, i * 2], [np.min(data[i][1]), np.max(data[i][1])], color=cmap(norm(i)), linestyle='-', linewidth=1, alpha=0.7)

# 在图的中间添加颜色块和组号
for i in range(group_count):
    group_center = i * 2
    width = 2  # Width of the color patches

    # 在中间添加颜色块和组号
    color_patch = plt.Rectangle((group_center - width / 2, -a_value), width, a_value * 2, color=cmap(norm(i)), alpha=0.5)
    ax_upper.add_patch(color_patch)
    ax_upper.text(group_center, -a_value * 0.5, f'Group {i + 1}A', ha='center', va='center', color='black')

    color_patch = plt.Rectangle((group_center - width / 2, -a_value), width, a_value * 2, color=cmap(norm(i)), alpha=0.5)
    ax_lower.add_patch(color_patch)
    ax_lower.text(group_center, -a_value * 0.5, f'Group {i + 1}B', ha='center', va='center', color='black')

# 添加图例到图右方
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(i)), markersize=10) for i in range(group_count)]
ax_upper.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend', labels=[f'Group {i + 1}' for i in range(group_count)])

# 调整上下部分的y轴位置
ax_upper.set_ylim(-a_value, a_value)
ax_lower.set_ylim(-a_value, a_value)

# 设置坐标轴和标题
ax_lower.set_xticks(np.arange(0, group_count * 2, 2))
ax_lower.set_xticklabels([str(i) for i in range(1, group_count + 1)])
ax_lower.set_xlabel('Group')

# 设置左侧纵轴
# 设置上半部分纵轴标签
ax_upper.yaxis.tick_left()
ax_upper.yaxis.set_label_position('left')
ax_upper.set_yticks(np.linspace(-a_value, a_value, 7))
ax_upper.set_yticklabels([f'{val:.0f}' for val in np.linspace(a_value, -a_value, 7)])
ax_upper.spines['left'].set_position(('outward', 10))  # Move the y-axis to the left
ax_upper.tick_params(axis='y', length=5, direction='out')  # Add tick lines to the y-axis

# 设置下半部分纵轴标签
ax_lower.yaxis.tick_left()
ax_lower.yaxis.set_label_position('left')
ax_lower.set_yticks(np.linspace(-a_value, a_value, 7))
ax_lower.set_yticklabels([f'{val:.0f}' for val in np.linspace(a_value, -a_value, 7)])
ax_lower.spines['left'].set_position(('outward', 10))  # Move the y-axis to the left
ax_lower.tick_params(axis='y', length=5, direction='out')  # Add tick lines to the y-axis

# 移除上半部分的框线和刻度
ax_upper.spines['top'].set_visible(False)
ax_upper.spines['right'].set_visible(False)
ax_upper.spines['bottom'].set_visible(False)

# 移除下半部分的框线和刻度
ax_lower.spines['top'].set_visible(False)
ax_lower.spines['right'].set_visible(False)
ax_lower.spines['bottom'].set_visible(False)

# 移除 x 轴刻度线
ax_upper.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax_lower.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# 显示网格
ax_upper.grid(False)
ax_lower.grid(False)

# 添加标题
ax_upper.set_title('Title')

# 保存图表为SVG文件
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')
plt.show()
