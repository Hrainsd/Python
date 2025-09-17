# 风玫瑰图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart3_mei_gui_tu\1.csv"
df = pd.read_csv(file_path)

# 使用数据框中的数据
categories = df['category']
numbers = df['number']

# 将月份转换为角度
theta = np.linspace(0.0, 2 * np.pi, len(categories), endpoint=False)

# 创建极坐标系
ax = plt.subplot(111, projection='polar')

# 绘制玫瑰图
bars = ax.bar(theta, numbers, align='center', alpha=0.75)

# 设置每个扇形的颜色
for bar in bars:
    bar.set_facecolor(plt.cm.jet(np.random.rand()))

# 设置角度标签
ax.set_xticks(theta)
ax.set_xticklabels(categories)

# 显示图形
plt.show()



# 玫瑰图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart3_mei_gui_tu\1.csv"
df = pd.read_csv(file_path)

# 使用数据框中的数据
categories = df['category']
numbers = df['number']

# 将月份转换为角度
theta = np.linspace(0.0, 2 * np.pi, len(categories), endpoint=False)

# 创建极坐标系
ax = plt.subplot(111, projection='polar')

# 绘制玫瑰图
bars = ax.bar(theta, numbers, align='center', alpha=0.75)

# 设置每个扇形的颜色
for bar in bars:
    bar.set_facecolor(plt.cm.jet(np.random.rand()))

# 关闭坐标轴，即移除环形线框
ax.set_axis_off()

# 显示图形
plt.show()



# 带数值的玫瑰图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从CSV文件读取数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart3_mei_gui_tu\1.csv"
df = pd.read_csv(file_path)

# 使用数据框中的数据
categories = df['category']
numbers = df['number']

# 将月份转换为角度
theta = np.linspace(0.0, 2 * np.pi, len(categories), endpoint=False)

# 创建极坐标系
ax = plt.subplot(111, projection='polar')

# 绘制玫瑰图
bars = ax.bar(theta, numbers, align='center', alpha=0.75)

# 在每个扇形上显示数值和添加颜色
for i, (bar, category) in enumerate(zip(bars, categories)):
    height = bar.get_height()
    color = plt.cm.jet(np.random.rand())  # Adding color to each bar
    bar.set_facecolor(color)
    ax.text(bar.get_x() + bar.get_width() / 2, height * 0.5, f'{height:.2f}', ha='center', va='center', color='white')

# 手动创建图例
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=bar.get_facecolor(), markersize=10) for bar in bars]
ax.legend(legend_handles, categories, loc='upper right', bbox_to_anchor=(1.3, 1))

# 添加标题
ax.set_title('Rose Chart', va='bottom', fontsize=16)

# 关闭坐标轴，即移除环形线框
ax.set_axis_off()

# 保存图表为SVG文件
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
