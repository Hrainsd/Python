# 敏感性分析---折线图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Sensitivity_Analysis\折线图_敏感性分析.csv"
df = pd.read_csv(file_path)

# 计算差值百分比
df['Difference (%)'] = ((df['number1'] - df['number2']) / df['number2']) * 100

# 创建绘图对象
fig, ax1 = plt.subplots()

# 绘制 number1 和 number2 的折线图
ax1.plot(df['category'], df['number1'], color='#FFC6CC', marker='', label='number1')
ax1.plot(df['category'], df['number2'], color='#E4C1F9', marker='', label='number2')
ax1.set_xlabel('Category', rotation= 0)
ax1.set_ylabel('Value')
ax1.tick_params(axis='x')

# 设置 x 轴刻度位置和标签
n = 4  # 显示五个标签
tick_positions = range(0, len(df['category']), len(df['category']) // (n-1))
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(df['category'][::len(df['category']) // (n-1)], rotation= 0)

# 添加右侧y轴
ax2 = ax1.twinx()
ax2.plot(df['category'], df['Difference (%)'], color='#7FFFD4', marker='o', label='Difference (%)')
ax2.set_ylabel('Difference (%)')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1, 1.2), frameon=False)

plt.title('Line Chart')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑

# 保存图表为SVG文件
plt.savefig('折线图_敏感性分析.svg', format='svg', bbox_inches='tight')

plt.show()



# 敏感性分析---条形图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Sensitivity_Analysis\条形图_敏感性分析.csv"
df = pd.read_csv(file_path)

# 提取数据
categories = df['category'].tolist()
number1 = df['number1'].tolist()
number2 = df['number2'].tolist()

# 按照输入文件的顺序排序
sorted_categories = sorted(categories, key=lambda x: categories.index(x), reverse=True)

# 为每个类别选择不同的颜色
colors = sns.color_palette('husl', n_colors=len(categories))

# 创建横向条形图
bar_width = 0.35  # 调整条形的宽度
bar_positions1 = range(len(sorted_categories))
bar_positions2 = [x + bar_width for x in bar_positions1]

# 计算number1和number2的差值百分比
difference_percentage = [abs((n2 - n1) / n1) * 100 for n1, n2 in zip(number1, number2)]

# 绘制条形图并设置颜色
bars1 = plt.barh(bar_positions1, number1, height=bar_width, color='lightblue', edgecolor='black', label='Number 1')
bars2 = plt.barh(bar_positions2, number2, height=bar_width, color='lightgreen', edgecolor='black', label='Number 2')

# 添加数值标签，并设置颜色
for bars, positions, values, color, category in zip([bars1, bars2], [bar_positions1, bar_positions2], [number1, number2], colors, sorted_categories):
    for bar, position, value in zip(bars, positions, values):
        plt.text(value + 0.1, position, str(value), va='center', color=color)

# 设置y轴刻度和标签
yticks = plt.yticks([pos + bar_width / 2 for pos in range(len(sorted_categories))], sorted_categories)

# 设置y轴刻度标签的颜色
for label, color in zip(yticks[1], colors):
    label.set_color(color)

plt.xlabel('Number of Genes', labelpad=10, fontsize=10)
plt.ylabel('KEGG Pathway Categories', rotation=90, labelpad=50, ha='right')

# 获取当前 y 轴标签
ylabel = plt.gca().get_ylabel()

# 设置标签位置
plt.gca().yaxis.set_label_coords(-0.1, 0.7)  # 设置 y 轴标签的位置

# 添加图例
plt.legend(bbox_to_anchor=(1, 0.9), frameon=False, fontsize=6)

# 移除网格线
plt.grid(False)

# 设置背景颜色
plt.gca().set_facecolor('white')

# 添加第二个坐标轴
plt.twiny()

# 绘制折线图
plt.plot(difference_percentage, range(len(sorted_categories)), color='#7FFFD4', marker='o', label='Difference Percentage')

# 设置顶部 x 轴刻度
x_ticks = [-20, -10, 0, 10, 20]  # 自定义刻度，根据实际情况调整
plt.xticks(x_ticks, [f"{tick}%" for tick in x_ticks])

# 设置标签
plt.xlabel('Difference Percentage', labelpad=10, fontsize=10)

plt.legend(bbox_to_anchor=(1, 0.8), frameon=False, fontsize=6)

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑

# 保存图表为SVG文件
plt.savefig('条形图_敏感性分析.svg', format='svg', bbox_inches='tight')

plt.show()



# 敏感性分析---雷达图
# model1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件并创建DataFrame
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Sensitivity_Analysis\雷达图_敏感性分析.csv"
df = pd.read_csv(file_path)

# 提取标签名称和number1、number2列的数据
labels = df['category']
number1_data = df['number1']
number2_data = df['number2']

# 设置雷达图的角度和数据
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
number1_values = number1_data.tolist()
number2_values = number2_data.tolist()

# 为了使雷达图闭合，再次添加第一个数据到最后
number1_values += number1_values[:1]
number2_values += number2_values[:1]
angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, number1_values, color='blue', alpha=0.25)
ax.fill(angles, number2_values, color='green', alpha=0.25)

# 添加标签，适当调整字体大小和角度
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=3)

# 设置雷达图的半径刻度
max_value = max(max(number1_values), max(number2_values))
tick_values = np.linspace(0, max_value, num=5)  # 设置5个刻度
ax.set_yticks(tick_values)

# 自定义刻度标签的颜色
tick_labels = [f'{value:.0f}' for value in tick_values]
ax.set_yticklabels(tick_labels, color='#4FC3F7', fontsize=5)  # 设置颜色和字体大小

# 关闭雷达图的径向网格线，保留角度网格线
ax.xaxis.grid(False)

# 显示图例
ax.legend(['number1', 'number2'], bbox_to_anchor=(1, 0.8), frameon=False)

plt.title('Radar Chart')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑

# 保存图表为SVG文件
plt.savefig('雷达图_敏感性分析_1.svg', format='svg', bbox_inches='tight')

plt.show()

# model2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件并创建DataFrame
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Sensitivity_Analysis\条形图_敏感性分析.csv"
df = pd.read_csv(file_path)

# 提取标签名称和number1、number2列的数据
labels = df['category']
number1_data = df['number1']
number2_data = df['number2']

# 设置雷达图的角度和数据
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
number1_values = number1_data.tolist()
number2_values = number2_data.tolist()

# 为了使雷达图闭合，再次添加第一个数据到最后
number1_values += number1_values[:1]
number2_values += number2_values[:1]
angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, number1_values, color='blue', alpha=0.25)
ax.fill(angles, number2_values, color='green', alpha=0.25)

# 添加标签
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)

# 设置雷达图的半径刻度
max_value = max(max(number1_values), max(number2_values))
tick_values = np.linspace(0, max_value, num=6)  # 设置5个刻度
ax.set_yticks(tick_values)

# 自定义刻度标签的颜色
tick_labels = [f'{value:.0f}' for value in tick_values]
ax.set_yticklabels(tick_labels, color='#848484', fontsize=5)  # 设置颜色和字体大小

# 关闭雷达图的径向网格线，保留角度网格线
ax.xaxis.grid(False)

# 显示图例
ax.legend(['number1', 'number2'], bbox_to_anchor=(1.2, 0.8), frameon=False)

plt.title('Radar Chart')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑

# 保存图表为SVG文件
plt.savefig('雷达图_敏感性分析_2.svg', format='svg', bbox_inches='tight')

plt.show()
