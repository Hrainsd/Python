# 嵌套箱线图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart20_qian_tao_zhu_zhuang_tu\111.csv"
df = pd.read_csv(file_path)

# 提取每天的a组和b组数据
days = ["day1", "day2", "day3", "day4", "day5", "day6"]

# 绘制箱线图
fig, ax = plt.subplots()

# 循环处理每一天的数据
for i, day in enumerate(days):
    # 提取a组和b组的数据
    a_data = df[f"{day}_a"]
    b_data = df[f"{day}_b"]

    # 绘制箱线图，共用一个x轴位置，并设置颜色
    a_box = ax.boxplot([a_data], positions=[2 * i + 1], widths=0.6, patch_artist=True, boxprops=dict(facecolor='#F67280'))
    b_box = ax.boxplot([b_data], positions=[2 * i + 1], widths=0.6, patch_artist=True, boxprops=dict(facecolor='#F8B195'))

# 设置图表标题和轴标签
plt.title("Boxplot of A and B groups for each day")
plt.xlabel("Days")
plt.ylabel("Values")

# 设置x轴刻度和标签
ax.set_xticks(range(1, len(days) * 2, 2))
ax.set_xticklabels(days)

# 添加图例
ax.legend([a_box["boxes"][0], b_box["boxes"][0]], ['A Group', 'B Group'], loc='upper left', frameon=False, bbox_to_anchor=(1, 1))

plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()



# 嵌套条形图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart20_qian_tao_zhu_zhuang_tu\111.csv"
df = pd.read_csv(file_path)

# 提取每天的a组和b组数据
days = ["day1", "day2", "day3", "day4", "day5", "day6"]

# 绘制条形图
fig, ax = plt.subplots()

# 循环处理每一天的数据
for i, day in enumerate(days):
    # 提取a组和b组的数据
    a_data = df[f"{day}_a"]
    b_data = df[f"{day}_b"]

    # 计算误差（可以根据实际情况调整）
    a_error = a_data.std()  # 使用标准差作为误差
    b_error = b_data.std()

    # 绘制条形图，并添加误差线
    ax.bar(2 * i + 1, a_data.mean(), yerr=a_error, color='#66CCCC', width=1, label='A Group', capsize=5)
    ax.bar(2 * i + 1, b_data.mean(), yerr=b_error, color='#336699', width=0.5, label='B Group', capsize=5)

# 设置图表标题和轴标签
plt.title("Bar chart of A and B groups for each day")
plt.xlabel("Days")
plt.ylabel("Values")

# 设置x轴刻度和标签
ax.set_xticks(range(1, len(days) * 2, 2))
ax.set_xticklabels(days)

# 添加图例，只包含A和B组
ax.legend(handles=[plt.Rectangle((0,0),1,1, color='#66CCCC', label='A Group'),
                   plt.Rectangle((0,0),1,1, color='#336699', label='B Group')],
          loc='upper left', frameon=False, bbox_to_anchor=(1.1, 1))

plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('output_plot_2.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()
