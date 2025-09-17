# 山峦图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart23_shan_luan_tu\1.csv"
df = pd.read_csv(file_path)

# 获取列名
columns = df.columns

# 设置画布
fig, ax = plt.subplots(figsize=(10, 6))

# 将数据转换为numpy数组
data = df.to_numpy().T

# 设置自定义颜色
colors = ['#76d7c3', '#a3e4d7', '#a2d9ce', '#a9dfbe', '#aaebc6', '#aed6f1', '#a9cce3', '#d2b4de', '#d7bce1', '#f5b6b1', '#e6b0aa', '#d35400']

# 使用ax.stackplot绘制堆叠山峦图，并设置颜色
ax.stackplot(range(len(df)), *data, labels=columns, colors=colors)

# 添加标题和标签
ax.set_title('Stacked Area Chart')
ax.set_xlabel('data points')
ax.set_ylabel('values')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()



# 多山峦图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart23_shan_luan_tu\1.csv"
df = pd.read_csv(file_path)

# 获取列名
columns = df.columns

# 设置画布和子图网格
fig, axs = plt.subplots(nrows=len(columns), ncols=1, figsize=(6, 4), sharex=True)

# 定义颜色
colors = ['#66C7B4', '#A2D8DD', '#F7CAC9', '#FFF0F5', '#FFF1E0', '#FFC48C','#A2D8F4','#3FBCE9','#FF9F80','#FFC48C','#F9D9A9','#F6A19B']

# 循环绘制每个子图
for i, ax in enumerate(axs):
    ax.stackplot(range(len(df)), df[columns[i]], color=colors[i], labels=[columns[i]])

    # 调整y轴标签的大小
    ax.set_ylabel(columns[i], fontsize=5)
    ax.set_yticks([])  # 移除y轴刻度值

# 添加标题和x轴标签
fig.suptitle('Stacked Area Chart for Each Day')
axs[-1].set_xlabel('data points')

# 调整子图布局
plt.tight_layout()

# 添加legend
fig.legend(labels=columns, bbox_to_anchor=(1, 0.9), loc='upper left', frameon=False, prop={'size': 5})

# Save the chart as an SVG file
plt.savefig('output_plot_2.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
