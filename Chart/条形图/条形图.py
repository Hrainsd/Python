# 创建横向条形图
import matplotlib.pyplot as plt

# 提供的数据
categories = [
    'Metabolism',
    'Carbohydrate metabolism',
    'Energy metabolism',
    'Global and Overview',
    'Metabolism of cofactors and vitamins',
    'Lipid metabolism',
    'Metabolism of terpenoids and polyketides',
    'Biosynthesis of other secondary metabolites',
    'Amino acid metabolism',
    'Xenobiotics biodegradation and metabolism'
]
gene_counts = [23, 23, 21, 22, 6, 9, 5, 6, 10, 2]

plt.barh(categories, gene_counts, color='skyblue', edgecolor='black')

# 交换x轴和y轴的标签
plt.xlabel('Number of Genes')
plt.ylabel('KEGG Pathway Categories')

# 添加图例
plt.legend(['Genes Number'])

# 移除网格线
plt.grid(False)

# 设置背景颜色
plt.gca().set_facecolor('white')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑
plt.show()



# 创建带数值的横向条形图
import matplotlib.pyplot as plt

# 提供的数据
categories = [
    'Metabolism',
    'Carbohydrate metabolism',
    'Energy metabolism',
    'Global and Overview',
    'Metabolism of cofactors and vitamins',
    'Lipid metabolism',
    'Metabolism of terpenoids and polyketides',
    'Biosynthesis of other secondary metabolites',
    'Amino acid metabolism',
    'Xenobiotics biodegradation and metabolism'
]
gene_counts = [23, 23, 21, 22, 6, 9, 5, 6, 10, 2]

bar_width = 0.8  # Adjust the width of the bars
bar_positions = range(len(categories))

plt.barh(bar_positions, gene_counts, height=bar_width, color='skyblue', edgecolor='black')

# 添加数值标签
for position, value in zip(bar_positions, gene_counts):
    plt.text(value + 0.1, position, str(value), va='center')

# 设置y轴刻度和标签
plt.yticks(bar_positions, categories)

# 交换x轴和y轴的标签
plt.xlabel('Number of Genes')
plt.ylabel('KEGG Pathway Categories')

# 添加图例
plt.legend(['Genes Number'])

# 移除网格线
plt.grid(False)

# 设置背景颜色
plt.gca().set_facecolor('white')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑
plt.show()



# 按顺序创建横向条形图
import matplotlib.pyplot as plt

# 提供的数据
categories = [
    'Metabolism',
    'Carbohydrate metabolism',
    'Energy metabolism',
    'Global and Overview',
    'Metabolism of cofactors and vitamins',
    'Lipid metabolism',
    'Metabolism of terpenoids and polyketides',
    'Biosynthesis of other secondary metabolites',
    'Amino acid metabolism',
    'Xenobiotics biodegradation and metabolism'
]
gene_counts = [23, 23, 21, 22, 6, 9, 5, 6, 10, 2]

# 按照输入文件的顺序排序
sorted_categories = sorted(categories, key=lambda x: categories.index(x), reverse=True)

bar_width = 0.8
bar_positions = range(len(sorted_categories))

plt.barh(bar_positions, gene_counts, height=bar_width, color='skyblue', edgecolor='black')

# 添加数值标签
for position, value in zip(bar_positions, gene_counts):
    plt.text(value + 0.1, position, str(value), va='center')

# 设置y轴刻度和标签
plt.yticks(bar_positions, sorted_categories)

# 交换x轴和y轴的标签
plt.xlabel('Number of Genes')
plt.ylabel('')

# 移动y轴标签到右边
plt.gca().yaxis.set_label_coords(0.7, 1)

# 调整y轴标签位置
plt.ylabel('KEGG Pathway Categories', rotation=0, labelpad=50, ha='right')

# 添加图例
plt.legend(['Genes Number'])

# 移除网格线
plt.grid(False)

# 设置背景颜色
plt.gca().set_facecolor('white')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑
plt.show()



# 使用csv文件按顺序创建横向条形图
import matplotlib.pyplot as plt
import pandas as pd

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart1_tiao_xing_tu\1.csv"
df = pd.read_csv(file_path)

# 提取数据
categories = df['category'].tolist()
gene_counts = df['number'].tolist()

# 按照输入文件的顺序排序
sorted_categories = sorted(categories, key=lambda x: categories.index(x), reverse=True)

# 创建横向条形图
bar_width = 0.8  # Adjust the width of the bars
bar_positions = range(len(sorted_categories))
plt.barh(bar_positions, gene_counts, height=bar_width, color='skyblue', edgecolor='black')

# 添加数值标签
for position, value in zip(bar_positions, gene_counts):
    plt.text(value + 0.1, position, str(value), va='center')

# 设置y轴刻度和标签
plt.yticks(bar_positions, sorted_categories)

# 交换x轴和y轴的标签
plt.xlabel('Number of Genes')
plt.ylabel('')

# 移动y轴标签到右边
plt.gca().yaxis.set_label_coords(0.6, 1)

# 调整y轴标签位置
plt.ylabel('KEGG Pathway Categories', rotation=0, labelpad=50, ha='right')

# 添加图例
plt.legend(['Genes Number'])

# 移除网格线
plt.grid(False)

# 设置背景颜色
plt.gca().set_facecolor('white')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑
plt.show()



# 使用csv文件按顺序创建不同颜色的横向条形图
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart1_tiao_xing_tu\1.csv"
df = pd.read_csv(file_path)

# 提取数据
categories = df['category'].tolist()
gene_counts = df['number'].tolist()

# 按照输入文件的顺序排序
sorted_categories = sorted(categories, key=lambda x: categories.index(x), reverse=True)

# 为每个类别选择不同的颜色
colors = sns.color_palette('husl', n_colors=len(categories))

# 创建横向条形图
bar_width = 0.8  # Adjust the width of the bars
bar_positions = range(len(sorted_categories))

# 绘制条形图并设置颜色
bars = plt.barh(bar_positions, gene_counts, height=bar_width, color=colors, edgecolor='black')

# 添加数值标签，并设置颜色
for bar, position, value, color in zip(bars, bar_positions, gene_counts, colors):
    plt.text(value + 0.1, position, str(value), va='center', color=color)

# 设置y轴刻度和标签
plt.yticks(bar_positions, sorted_categories)

# 交换x轴和y轴的标签
plt.xlabel('Number of Genes')
plt.ylabel('')

# 移动y轴标签到右边
plt.gca().yaxis.set_label_coords(0.6, 1)

# 调整y轴标签位置
plt.ylabel('KEGG Pathway Categories', rotation=0, labelpad=50, ha='right')

# 添加图例
plt.legend(['Genes Number'])

# 移除网格线
plt.grid(False)

# 设置背景颜色
plt.gca().set_facecolor('white')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑
plt.show()



# 使用csv文件按顺序创建横向条形图
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart1_tiao_xing_tu\1.csv"
df = pd.read_csv(file_path)

# 提取数据
categories = df['category'].tolist()
gene_counts = df['number'].tolist()

# 按照输入文件的顺序排序
sorted_categories = sorted(categories, key=lambda x: categories.index(x), reverse=True)

# 为每个类别选择不同的颜色
colors = sns.color_palette('husl', n_colors=len(categories))

# 创建横向条形图
bar_width = 0.8  # Adjust the width of the bars
bar_positions = range(len(sorted_categories))

# 绘制条形图并设置颜色
bars = plt.barh(bar_positions, gene_counts, height=bar_width, color=colors, edgecolor='black')

# 添加数值标签，并设置颜色
for bar, position, value, color, category in zip(bars, bar_positions, gene_counts, colors, sorted_categories):
    plt.text(value + 0.1, position, str(value), va='center', color=color)

# 设置y轴刻度和标签
yticks = plt.yticks(bar_positions, sorted_categories)

# 设置y轴刻度标签的颜色
for label, color in zip(yticks[1], colors):
    label.set_color(color)

# 交换x轴和y轴的标签
plt.xlabel('Number of Genes')
plt.ylabel('')

# 移动y轴标签到右边
plt.gca().yaxis.set_label_coords(0.6, 1)

# 调整y轴标签位置
plt.ylabel('KEGG Pathway Categories', rotation=0, labelpad=50, ha='right')

# 添加图例
plt.legend(['Genes Number'])

# 移除网格线
plt.grid(False)

# 设置背景颜色
plt.gca().set_facecolor('white')

# 保存图表为SVG文件
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.tight_layout()  # 调整子图参数，使布局更加紧凑
plt.show()
