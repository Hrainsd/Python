# 堆叠图
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from cycler import cycler
from collections import OrderedDict

# 设置字体为 Times New Roman
font_path = fm.findfont(fm.FontProperties(family='Times New Roman'))

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart15_dong_tai_dui_die_tu\1.csv"
df = pd.read_csv(file_path)

# 设置图形大小
fig, ax = plt.subplots(figsize=(15, 15))

# 获取数据列
data_columns = df.columns[1:]

# 计算环的宽度
width = 0.2

# 设置半径
radius_index1 = 0.4
radius_index2 = 0.85
radius_index3 = 1.25

# 添加标题并调整位置，同时设置字体大小
title_font_properties = fm.FontProperties(fname=font_path, size=28)
ax.set_title('Ring Chart', fontproperties=title_font_properties)
ax.title.set_position([0.5, 1.2])  # 调整标题的绝对位置

# 创建SciPy顶刊颜色集合
sci_colors = ["#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#FFD92F", "#B15928", "#7FC97F", "#CAB2D6", "#8C564B",
              "#BCBD22", "#17BECF", "#AEC7E8", "#FF9E4A", "#98DF8A", "#FFD92F", "#1F78B4", "#7DAEE0", "#B395BD", "#33A02C",
              "#6A3D9A", "#FF7F00", "#E31A1C", "#1F78B4", "#33A02C", "#B15928", "#7FC97F", "#CAB2D6", "#8C564B", "#BCBD22"]

# 创建Factor到颜色的映射字典
factor_color_mapping = OrderedDict(zip(df['factor'], sci_colors))

# 绘制第一个环形
wedges1, texts1, autotexts1 = ax.pie(df[data_columns[0]], labels=None, autopct='', startangle=0, counterclock=False,
                                     wedgeprops=dict(width=width, edgecolor='w'), colors=[factor_color_mapping[f] for f in df['factor']], radius=radius_index1)

# 绘制第二个环形
wedges2, texts2, autotexts2 = ax.pie(df[data_columns[1]], labels=None, autopct='', startangle=0, counterclock=False,
                                     wedgeprops=dict(width=width, edgecolor='w'), colors=[factor_color_mapping[f] for f in df['factor']], radius=radius_index2)

# 绘制第三个环形
wedges3, texts3, autotexts3 = ax.pie(df[data_columns[2]], labels=None, autopct='', startangle=0, counterclock=False,
                                     wedgeprops=dict(width=width, edgecolor='w'), colors=[factor_color_mapping[f] for f in df['factor']], radius=radius_index3)

# 添加标签
label_distance = 1.1  # 调整标签距离环的距离
ax.text(0, -radius_index1 * label_distance, 'Index1', ha='center', va='center', fontsize=20, fontweight='bold',
        fontproperties=fm.FontProperties(fname=font_path))
ax.text(0, -radius_index2 * label_distance, 'Index2', ha='center', va='center', fontsize=20, fontweight='bold',
        fontproperties=fm.FontProperties(fname=font_path))
ax.text(0, -radius_index3 * label_distance, 'Index3', ha='center', va='center', fontsize=20, fontweight='bold',
        fontproperties=fm.FontProperties(fname=font_path))

# 显示图例，设置ncol参数为2，并调整字体大小
ax.legend(df['factor'], title='Legend', loc='upper left', bbox_to_anchor=(1.1, 1), ncol=2, frameon=False,
          prop=fm.FontProperties(fname=font_path, size=24))

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
