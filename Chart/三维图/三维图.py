# 三维散点图
import pandas as pd
import plotly.express as px

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart25_san_wei_tu\1.csv"
df = pd.read_csv(file_path)

# 定义色阶
colors = ['#F7CAC9', '#FADBD8', '#F5EEF8', '#D2B4DE', '#A2D5F2', '#76B6EA']

# 绘制3D曲面图
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Z', color_continuous_scale=colors)

# 设置图表布局
fig.update_layout(scene=dict(zaxis=dict(range=[df['Z'].min(), df['Z'].max()])))

# 显示图表
fig.show()



# 三维散点图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart25_san_wei_tu\1.csv"
df = pd.read_csv(file_path)

# 创建三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义色阶
colors = ['#F7CAC9', '#FADBD8', '#F5EEF8', '#D2B4DE', '#A2D5F2', '#76B6EA']

# 绘制散点图，并指定色阶
scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Z'], cmap=LinearSegmentedColormap.from_list('mycmap', colors), marker='o')

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 调整坐标轴刻度线向内
ax.tick_params(axis='x', which='both', direction='in')
ax.tick_params(axis='y', which='both', direction='in')
ax.tick_params(axis='z', which='both', direction='in')

# 设置视角为45度
ax.view_init(elev=20, azim=60)

# 删除显示各个面的线和坐标轴内部的线，只保留外边框
ax.grid(False)
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# 设置坐标轴平面边框颜色为黑色
ax.w_xaxis.pane.set_edgecolor('k')
ax.w_yaxis.pane.set_edgecolor('k')
ax.w_zaxis.pane.set_edgecolor('k')

# 添加色阶条，并调整长度
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Z Value')  # 设置色阶条标签

# 调整子图布局
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('三维散点图.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()



# 三维气泡图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart25_san_wei_tu\1.csv"
df = pd.read_csv(file_path)

# 创建三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义色阶
colors = ['#F7CAC9', '#FADBD8', '#F5EEF8', '#D2B4DE', '#A2D5F2', '#76B6EA']

# 绘制气泡图，颜色和大小由Z值控制
scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Z'], cmap=LinearSegmentedColormap.from_list('mycmap', colors), s=df['Z']*100, marker='o', alpha=0.6)

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 调整坐标轴刻度线向内
ax.tick_params(axis='x', which='both', direction='in')
ax.tick_params(axis='y', which='both', direction='in')
ax.tick_params(axis='z', which='both', direction='in')

# 设置视角为45度
ax.view_init(elev=20, azim=60)

# 删除显示各个面的线和坐标轴内部的线，只保留外边框
ax.grid(False)
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# 设置坐标轴平面边框颜色为黑色
ax.w_xaxis.pane.set_edgecolor('k')
ax.w_yaxis.pane.set_edgecolor('k')
ax.w_zaxis.pane.set_edgecolor('k')

# 添加色阶条
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Z Value')  # 设置色阶条标签

# 调整子图布局
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('三维气泡图.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()



# 三维柱形图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart25_san_wei_tu\1.csv"
df = pd.read_csv(file_path)

# 创建三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义色阶
colors = ['#F7CAC9', '#FADBD8', '#F5EEF8', '#D2B4DE', '#A2D5F2', '#76B6EA']

# 循环绘制每个柱形
for i in range(len(df)):
    ax.bar3d(df['X'][i], df['Y'][i], 0, 0.1, 0.1, df['Z'][i], shade=True, color=colors[int(df['Z'][i] / df['Z'].max() * (len(colors) - 1))])

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 设置视角为45度
ax.view_init(elev=20, azim=60)

# 创建 ScalarMappable，用于创建颜色条
norm = Normalize(vmin=df['Z'].min(), vmax=df['Z'].max())
sm = ScalarMappable(cmap=LinearSegmentedColormap.from_list('mycmap', colors), norm=norm)
sm.set_array([])

# 删除显示各个面的线和坐标轴内部的线，只保留外边框
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# 设置坐标轴平面边框颜色为黑色
ax.w_xaxis.pane.set_edgecolor('k')
ax.w_yaxis.pane.set_edgecolor('k')
ax.w_zaxis.pane.set_edgecolor('k')

# 添加色阶条
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Z Value')  # 设置色阶条标签

# 调整子图布局
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('三维柱形图.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()



# 三维曲面图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart25_san_wei_tu\1.csv"
df = pd.read_csv(file_path)

# 定义色阶
colors = ['#F7CAC9', '#FADBD8', '#F5EEF8', '#D2B4DE', '#A2D5F2', '#76B6EA']

# 创建3D曲面图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成网格点
xi, yi = np.linspace(df['X'].min(), df['X'].max(), 100), np.linspace(df['Y'].min(), df['Y'].max(), 100)
xi, yi = np.meshgrid(xi, yi)

# 插值
zi = griddata((df['X'], df['Y']), df['Z'], (xi, yi), method='linear')

# 创建自定义颜色映射
custom_cmap = ListedColormap(colors)

# 绘制3D曲面
surface = ax.plot_surface(xi, yi, zi, cmap=custom_cmap, alpha=0.7)

# 设置视角为45度
ax.view_init(elev=20, azim=60)

# 创建 ScalarMappable，用于创建颜色条
norm = Normalize(vmin=df['Z'].min(), vmax=df['Z'].max())
sm = ScalarMappable(cmap=LinearSegmentedColormap.from_list('mycmap', colors), norm=norm)
sm.set_array([])

# 删除显示各个面的线和坐标轴内部的线，只保留外边框
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# 设置坐标轴平面边框颜色为黑色
ax.w_xaxis.pane.set_edgecolor('k')
ax.w_yaxis.pane.set_edgecolor('k')
ax.w_zaxis.pane.set_edgecolor('k')

# 添加色阶条
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Z Value')  # 设置色阶条标签

# 调整子图布局
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('三维曲面图.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()



# 三维切片图
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.tri import Triangulation
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart25_san_wei_tu\1.csv"
df = pd.read_csv(file_path)

# 提取X、Y、Z列数据
x = df['X']
y = df['Y']
z = df['Z']

# 定义色阶
colors = ['#F7CAC9', '#FADBD8', '#F5EEF8', '#D2B4DE', '#A2D5F2', '#76B6EA']

# 创建三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建三角网格
triang = Triangulation(x, y)

# 绘制三维切片图
plot = ax.tricontourf(triang, z, cmap=plt.cm.colors.ListedColormap(colors))

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置视角为45度
ax.view_init(elev=20, azim=60)

# 创建 ScalarMappable，用于创建颜色条
norm = Normalize(vmin=df['Z'].min(), vmax=df['Z'].max())
sm = ScalarMappable(cmap=LinearSegmentedColormap.from_list('mycmap', colors), norm=norm)
sm.set_array([])

# 删除显示各个面的线和坐标轴内部的线，只保留外边框
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# 设置坐标轴平面边框颜色为黑色
ax.w_xaxis.pane.set_edgecolor('k')
ax.w_yaxis.pane.set_edgecolor('k')
ax.w_zaxis.pane.set_edgecolor('k')

# 添加色阶条
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Z Value')  # 设置色阶条标签

# 调整子图布局
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('三维切片图_1.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()



# 三维切片图
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# 读取CSV文件
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\chart_table\chart\chart25_san_wei_tu\1.csv"
df = pd.read_csv(file_path)

# 提取X、Y、Z列数据
x = df['X']
y = df['Y']
z = df['Z']

# 定义色阶
colors = ['#F7CAC9', '#FADBD8', '#F5EEF8', '#D2B4DE', '#A2D5F2', '#76B6EA']

# 创建三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建 XY 平面的三角网格
triang_xy = Triangulation(x, y)

# 绘制 XY 平面的切片图
plot_xy = ax.tricontourf(triang_xy, z, cmap=plt.cm.colors.ListedColormap(colors), zdir='z')

# 创建 YZ 平面的三角网格
triang_yz = Triangulation(y, z)

# 绘制 YZ 平面的切片图
plot_yz = ax.tricontourf(triang_yz, x, cmap=plt.cm.colors.ListedColormap(colors), zdir='x')

# 创建 XZ 平面的三角网格
triang_xz = Triangulation(x, z)

# 绘制 XZ 平面的切片图
plot_xz = ax.tricontourf(triang_xz, y, cmap=plt.cm.colors.ListedColormap(colors), zdir='y')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置视角为45度
ax.view_init(elev=20, azim=60)

# 创建 ScalarMappable，用于创建颜色条
norm = Normalize(vmin=df['Z'].min(), vmax=df['Z'].max())
sm = ScalarMappable(cmap=LinearSegmentedColormap.from_list('mycmap', colors), norm=norm)
sm.set_array([])

# 删除显示各个面的线和坐标轴内部的线，只保留外边框
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# 设置坐标轴平面边框颜色为黑色
ax.w_xaxis.pane.set_edgecolor('k')
ax.w_yaxis.pane.set_edgecolor('k')
ax.w_zaxis.pane.set_edgecolor('k')

# 添加色阶条
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Z Value')  # 设置色阶条标签

# 调整子图布局
plt.tight_layout()

# Save the chart as an SVG file
plt.savefig('三维切片图_2.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()
