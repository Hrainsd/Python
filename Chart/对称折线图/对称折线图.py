# 纵向对称折线图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart14_dui_chen_zhe_xian_tu\1.csv"
df = pd.read_csv(file_path)

# 创建一个画布
fig = plt.figure(figsize=(10, 8))

# 设置上半部分轴的位置
ax1 = fig.add_axes([0.1, 0.60, 0.8, 0.3])  # [left, bottom, width, height]
line1, = ax1.plot(df['index1'], marker='*', label='降雨量', color='#EF767A')
ax1.set_ylabel('rainfall')
ax1.tick_params(axis='both', which='both', length=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])  # Clear x-axis ticks

# 在上半部分添加水平和垂直的浅色网格线
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 设置下半部分轴的位置
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3])  # [left, bottom, width, height]
line2, = ax2.plot(df['index2'], marker='o', label='蒸发量', color='#FFA500')
ax2.set_ylabel('evaporation')
ax2.invert_yaxis()
ax2.tick_params(axis='both', which='both', length=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_xticks([])  # Clear x-axis ticks

# 在下半部分添加水平和垂直的浅色网格线
ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 设置上半分X轴标签
ax3 = fig.add_axes([0.15, 0.55, 0.8, 0.05])  # [left, bottom, width, height]
factor_count = 8
factor_indices = range(0, len(df['factor']), len(df['factor']) // factor_count)
ax3.set_xticks(factor_indices)
ax3.set_xticklabels(df['factor'][factor_indices], rotation=0, ha='right')
ax3.tick_params(axis='both', which='both', length=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.set_yticks([])  # Remove y-axis labels

# Remove the last X-axis label for ax3
ax3.set_xticks(ax3.get_xticks()[:-1])

thresholds_upper = [0, 20, 50, 100]  # Example thresholds for upper part
colors_upper = ['#DBDB8D' , '#00BFFF', '#FFA500', '#FF0000']  # Blue, Orange, Red

for i in range(len(thresholds_upper) - 1):
    ax1.axhline(y=thresholds_upper[i], color=colors_upper[i], linestyle='-', linewidth=1)
    ax1.fill_between(df['factor'], thresholds_upper[i], thresholds_upper[i + 1], color=colors_upper[i], alpha=0.2)

# 设置下半分X轴标签
ax4 = fig.add_axes([0.15, 0.45, 0.8, 0.05])  # [left, bottom, width, height]
ax4.set_xticks(factor_indices)
ax4.set_xticklabels(df['factor'][factor_indices], rotation=0, ha='right')
ax4.tick_params(axis='both', which='both', length=0)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.set_yticks([])  # Remove y-axis labels

# Remove the last X-axis label for ax4
ax4.set_xticks(ax4.get_xticks()[:-1])

thresholds_lower = [0, 20, 50, 100]  # Example thresholds for lower part
colors_lower = ['#ABD0F1' , '#00FF00', '#FFFF00', '#800080']  # Green, Yellow, Purple

for i in range(len(thresholds_lower) - 1):
    ax2.axhline(y=thresholds_lower[i], color=colors_lower[i], linestyle='-', linewidth=1)
    ax2.fill_between(df['factor'], thresholds_lower[i], thresholds_lower[i + 1], color=colors_lower[i], alpha=0.2)

# 添加title和legend，legend不要边框
fig.suptitle('Plot', fontsize=16)  # Add title to the entire figure
# legend_labels = ['Rainfall', 'Evaporation']
# fig.legend([line1, line2], legend_labels, loc='upper right', bbox_to_anchor=(0.6, 0.55), frameon=False)

# Save the chart as an SVG file
plt.savefig('output_plot.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()



# 横向对称折线图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart14_dui_chen_zhe_xian_tu\1.csv"
df = pd.read_csv(file_path)

# 创建一个画布
fig = plt.figure(figsize=(10, 8))

# 设置左边轴的位置
ax1 = fig.add_axes([0.2, 0.1, 0.3, 0.8])  # [left, bottom, width, height]
bar1 = ax1.barh(df['index1'], df['factor'], color='#EF767A')
ax1.set_xlabel('rainfall')
ax1.tick_params(axis='both', which='both', length=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_yticks([])  # Clear x-axis ticks
ax1.set_xticks(range(0,101,25))
custom_labels_ax1 = ['0', '25', '50', '75','100']
ax1.set_xticklabels(custom_labels_ax1)  # Set custom y-axis labels

# 在左边添加垂直的浅色网格线
ax1.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

# 设置右边轴的位置
ax2 = fig.add_axes([0.5, 0.1, 0.3, 0.8])  # [left, bottom, width, height]
bar2 = ax2.barh(df['index2'], df['factor'], color='#FFA500')
ax2.set_xlabel('evaporation')
ax2.invert_xaxis()
ax2.tick_params(axis='both', which='both', length=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_yticks([])  # Clear x-axis ticks
ax2.set_xticks(range(0,101,25))
custom_labels_ax1 = ['0', '25', '50', '75','100']
ax2.set_xticklabels(custom_labels_ax1)  # Set custom y-axis labels

# 在右边添加垂直的浅色网格线
ax2.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

# 设置X轴标签
ax3 = fig.add_axes([0.1, 0.1, 0.1, 0.8])  # [left, bottom, width, height]
factor_count = 8
factor_indices = range(0, len(df['factor']), len(df['factor']) // factor_count)
ax3.set_yticks(factor_indices)
ax3.set_yticklabels(df['factor'][factor_indices], rotation=0, ha='right')
ax3.tick_params(axis='both', which='both', length=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.set_xticks([])  # Remove x-axis labels

# Remove the last Y-axis label for ax3
ax3.set_yticks(ax3.get_yticks()[:-1])

thresholds_upper = [0, 25, 50, 100]  # Example thresholds for upper part
colors_upper = ['#DBDB8D', '#00BFFF', '#FFA500', '#FF0000']  # Blue, Orange, Red

for i in range(len(thresholds_upper) - 1):
    ax1.axvline(x=thresholds_upper[i], color=colors_upper[i], linestyle='-', linewidth=1)
    ax1.fill_betweenx(df['factor'], thresholds_upper[i], thresholds_upper[i + 1], color=colors_upper[i], alpha=0.2)

# 设置下半分X轴标签
ax4 = fig.add_axes([0.9, 0.1, 0.1, 0.8])  # [left, bottom, width, height]
ax4.set_yticks(factor_indices)
ax4.set_yticklabels(df['factor'][factor_indices], rotation=0, ha='right')
ax4.tick_params(axis='both', which='both', length=0)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.set_xticks([])  # Remove x-axis labels

# Remove the last Y-axis label for ax4
ax4.set_yticks(ax4.get_yticks()[:-1])

thresholds_lower = [0, 25, 50, 100]  # Example thresholds for lower part
colors_lower = ['#ABD0F1', '#00FF00', '#FFFF00', '#800080']  # Green, Yellow, Purple

for i in range(len(thresholds_lower) - 1):
    ax2.axvline(x=thresholds_lower[i], color=colors_lower[i], linestyle='-', linewidth=1)
    ax2.fill_betweenx(df['factor'], thresholds_lower[i], thresholds_lower[i + 1], color=colors_lower[i], alpha=0.2)

# 设置legend，legend不要边框
fig.suptitle('Plot', fontsize=16)  # Add title to the entire figure

plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('output_plot_rotated.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()
