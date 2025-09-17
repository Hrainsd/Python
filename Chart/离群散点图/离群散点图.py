# 离群散点图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart12_li_qun_san_dian_tu\1.csv"
df = pd.read_csv(file_path)

# 按组绘制散点图
groups = df.groupby('group')
fig, ax = plt.subplots()

# 定义彩色
colors = {'group1': 'red', 'group2': 'blue', 'group3': 'green'}  # 根据实际情况补充颜色

for name, group in groups:
    # 绘制散点图
    ax.scatter(group['index1'], group['index2'], label=name)

    # 根据组别画大圆圈
    group_center = (group['index1'].mean(), group['index2'].mean())
    group_radius = max(group['index1'].max() - group['index1'].mean(), group['index2'].max() - group['index2'].mean())
    group_circle = plt.Circle(group_center, group_radius, color=colors[name], alpha=0.3, label=f'{name} Cir')
    ax.add_patch(group_circle)

# 添加图例，并将图例放在右边，去除图例的边框
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# 添加轴标签和标题
ax.set_xlabel('index1')
ax.set_ylabel('index2')
ax.set_title('Scatter Plot by Group with Group Circles')

# 去除上方和右方的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
