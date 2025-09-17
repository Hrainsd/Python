# 双坐标轴图(箱线图+折线图)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart13_shuang_zuo_biao_zhou_tu\1.csv"
df = pd.read_csv(file_path)

# 提取需要绘制的列
boxplot_columns = ['rz_index1', 'vnux_index1', 'bsi_index1', 'mwf_index1', 'shku_index1', 'hmnn_index1']
lineplot_columns = ['rz_index2', 'vnux_index2', 'bsi_index2', 'mwf_index2', 'shku_index2', 'hmnn_index2']

# 颜色
colors = ['#FF5252', '#FFD740', '#4CAF50', '#2196F3', '#FF4081', '#536DFE']

# 创建图形
fig, ax1 = plt.subplots(figsize=(15, 8))

# 绘制箱型图
sns.boxplot(data=df[boxplot_columns], palette=colors, ax=ax1)
ax1.set_ylabel('Index1 Values')
ax1.set_title('Boxplot and Lineplot')

# 设置 x 轴标签为类别名称，并旋转 x 轴标签
ax1.set_xticklabels(['rz', 'vnux', 'bsi', 'mwf', 'shku', 'hmnn'])
ax1.tick_params(axis='x')

# 计算index2每类数据的平均值
avg_values = df[lineplot_columns].mean()

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制折线图（使用平均值）
ax2.plot(avg_values, label='Index2 Mean', marker='o', linestyle='-', color="#c77cff", linewidth=2, markersize=8)
ax2.set_ylabel('Index2 Values', rotation=-90, labelpad=20)  # Adjust labelpad as needed

# 添加箱线图的legend
legend_labels = ['rz', 'vnux', 'bsi', 'mwf', 'shku', 'hmnn']
legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(legend_labels))]
legend = ax1.legend(legend_handles, legend_labels, title='Legend', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)

# 去除上方的边框
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 调整图形布局，使得 legend 不会被遮挡
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()



# 双坐标轴图(柱形图+折线图)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart13_shuang_zuo_biao_zhou_tu\1.csv"
df = pd.read_csv(file_path)

# 提取需要绘制的列
barplot_columns = ['rz_index1', 'vnux_index1', 'bsi_index1', 'mwf_index1', 'shku_index1', 'hmnn_index1']
lineplot_columns = ['rz_index2', 'vnux_index2', 'bsi_index2', 'mwf_index2', 'shku_index2', 'hmnn_index2']

# 颜色
colors = ['#FF5252', '#FFD740', '#4CAF50', '#2196F3', '#FF4081', '#536DFE']

# 创建图形
fig, ax1 = plt.subplots(figsize=(15, 8))

# 绘制柱形图
for i, col in enumerate(barplot_columns):
    x = i
    y = df[col].mean()
    error = df[col].std()  # Assuming you want to use standard deviation for error bars

    plt.bar(x, y, color=colors[i], width=0.6, edgecolor="#FF99CC", alpha=0.7, label=f'{col} Values')
    plt.errorbar(x, y, yerr=error, fmt='none', ecolor='black', capsize=5)

ax1.set_ylabel('Index1 Values')
ax1.set_title('Barplot and Lineplot')

# 设置 x 轴标签为类别名称
ax1.set_xticks(range(len(barplot_columns)))
ax1.set_xticklabels(['rz', 'vnux', 'bsi', 'mwf', 'shku', 'hmnn'])

# 计算index2每类数据的平均值
avg_values = df[lineplot_columns].mean()

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制折线图（使用平均值）
ax2.plot(avg_values, label='Index2 Mean', marker='o', linestyle='-', color="#c77cff", linewidth=2, markersize=8)
ax2.set_ylabel('Index2 Values', rotation=-90, labelpad=20)  # Adjust labelpad as needed

# 添加柱形图的legend
legend_labels = barplot_columns
legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=2, alpha=0.7) for i in range(len(legend_labels))]
legend_handles.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Error Bar'))
legend = ax1.legend(legend_handles, legend_labels, title='Legend', loc='upper left',
                    bbox_to_anchor=(1.05, 1), frameon=False)

# 去除上方的边框
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 调整图形布局，使得 legend 不会被遮挡
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('output_plot2.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()
