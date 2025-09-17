import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Prophet\1.csv"
df = pd.read_csv(file_path)

# 重命名列以符合Prophet的要求
df = df.rename(columns={'DS': 'ds', 'Y': 'y'})

# 创建Prophet模型
model = Prophet()

# 拟合模型
model.fit(df)

# 创建一个数据框，包含未来要预测的日期
future = model.make_future_dataframe(periods=400)  # 假设预测未来400天

# 进行预测
forecast = model.predict(future)

# 设置预测结果的颜色
fig = model.plot(forecast)
ax = fig.gca()
ax.get_lines()[0].set_color('#6C5B7B')  # 设置线的颜色
ax.fill_between(ax.get_lines()[0].get_xdata(),
                ax.get_lines()[0].get_ydata(), color='#C06C84', alpha=0.2)  # 设置填充区域的颜色

# 调整图布局
plt.tight_layout()

# 保存图表为SVG文件
fig.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Prophet\forecast_chart.svg")

# 显示图表
plt.show()

# 绘制成分图
fig2 = model.plot_components(forecast)

# 设置成分图的颜色
for ax in fig2.get_axes():
    for line in ax.get_lines():
        line.set_color('#61C0BF')  # 设置线的颜色

# 调整图布局
plt.tight_layout()

# 保存成分图为SVG文件
fig2.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Prophet\component_chart.svg")

# 显示成分图
plt.show()

# 将原始数据和预测结果合并
merged_df = pd.concat([df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]], axis=1)

# 保存合并后的数据框为CSV文件
merged_df.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Prophet\2.csv", index=False)

# 显示合并后的数据框
print("Merged Data:")
print(merged_df.head())

# 计算差值百分比与均值并保存到3.csv
merged_df['y_diff_percentage'] = ((merged_df['yhat'] - merged_df['y']) / merged_df['y']) * 100

result_df = pd.DataFrame({
    'y_diff_percentage': merged_df['y_diff_percentage']
})

result_df.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\Prophet\3.csv", index=False)

# 显示计算结果
print("Calculation Results:")
print(result_df)
print("Mean of y_diff_percentage:", merged_df['y_diff_percentage'].mean())
