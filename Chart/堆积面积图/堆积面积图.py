# 堆积面积图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 读取 CSV 文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart24_dui_ji_mian_ji_tu\1.csv"
df = pd.read_csv(file_path)

# 提取数据
salinity_values = df['Salinity']
num_groups = len(df.columns) - 1  # 除了盐度列外的其他列数

# 提取相对丰度数据
relative_abundance_data = [df[f'Group_{i}_Relative_Abundance'] for i in range(1, num_groups + 1)]

# 指定颜色
colors = ['#9b59b6', '#8d44ad', '#2980b9', '#3398da', '#1abc9b', '#169f85']

# 绘制堆积面积图
plt.figure(figsize=(10, 6))

for i in range(num_groups):
    if i == 0:
        plt.fill_between(salinity_values, 0, relative_abundance_data[i], label=f'Group {i+1}', color=colors[i])
    else:
        plt.fill_between(salinity_values, np.sum(relative_abundance_data[:i], axis=0),
                         np.sum(relative_abundance_data[:i+1], axis=0), label=f'Group {i+1}', color=colors[i])

# 添加标签和标题
plt.xlabel('Salinity')
plt.ylabel('Relative Abundance (%)')
plt.title('Stacked Area Plot: Salinity vs Relative Abundance for Six Groups')

# 创建图例并指定图例项为线的形式，代表圆形
legend_elements = [Line2D([0], [0], marker='*', color='w', markerfacecolor=colors[i], markersize=10, label=f'Group {i+1}') for i in range(num_groups)]
# markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

# 调整子图布局
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()
