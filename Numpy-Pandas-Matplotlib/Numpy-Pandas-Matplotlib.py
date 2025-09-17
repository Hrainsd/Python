### numpy
import random
import numpy as np
import pandas as pd

array = np.array([[1,3],[2,4],[2,6]])
print(array)
print('number of dim:',array.ndim)
print('shape:',array.shape)
print('size:',array.size)

a = np.array([1,23,4])
print(a)
b = np.zeros((3,4))
print(b)
c = np.ones((3,4))
print(c)
d = np.empty((4,4))
print(d)
e = np.arange(1,10,2) # 取不到10
print(e)
ee = np.arange(1,21).reshape((5,4))
print(ee)
f = np.linspace(1,20,8).reshape((2,4))
print(f)

a1 = np.arange(4)  # [0,1,2,3]
print(a1)
a2 = np.random.randn(5) # randn返回标准正态分布的一行数据
print(a2)
b1 = np.array([12,14,16,18])
c1 = b1-a1
print(c1)
print(10*np.sin(a1))
print(b1<15)
a11 = np.array([[1,4],[2,3]])
b11 = np.array([[1,5],[6,7]])
c2 = a11*b11 # 对应元素相乘
c3 = np.dot(a11,b11) # 矩阵的乘法1
c33 = a11.dot(b11) # 矩阵的乘法2
print(c2)
print(c3)
print(c33)

# 索引从零开始，零行零列
ab = np.random.random((2,4))
print(ab)
print(np.sum(ab,axis = 0)) # 0代表列
print(np.max(ab,axis = 1)) # 1代表行
print(np.min(ab)) # 不加，则在所有元素中找
print(np.median(ab)) # 中位数
print(np.mean(ab)) # 平均值
print(np.average(ab))

ac = np.arange(2,14).reshape(3,4)
print(np.argmin(ac)) # 最小值的索引，0
print(np.argmax(ac)) # 最大值的索引，11
print(np.cumsum(ac))  # 累加，从第一个数按行累加
print(np.diff(ac)) # 左右相邻两个数的差
ac1 = np.array([[17,47,39],[45,32,31]])
print(np.sort(ac1)) # 逐行从小到大排序
print(np.nonzero(ac1)) # 输出行与列的索引值
print(ac1.T) # 矩阵的转置
print(np.transpose(ac1))
print(np.clip(ac1,20,40)) # 保留之间的数，小于20的全变为20，大于40的全变为40

ad = np.arange(1,15)
ad1 = np.arange(1,15).reshape(7,2)
print(ad[2]) # 第2个元素
print(ad1[6]) # 第6行元素
print(ad1[1][1]) # 第1行第1列的元素
print(ad1[1,1])
print(ad1[0,:]) # 第0行的所有元素
print(ad1[:,1]) # 第1列的所有元素
print(ad1[1:3,:]) # 取不到第3行
for cell in ad1:
    print(cell)
for cell1 in ad1.T:
    print(cell1)
print(ad1.flatten()) # x.flatten()将矩阵x变为1行
for item in ad1.flat: # 打印出每个元素
    print(item,end=" ")
    print()

ae = np.array([1,2,3])
ae1 = np.array([3,4,5])
ae2 = np.vstack((ae,ae1)) # 竖直合并
ae3 = np.hstack((ae,ae1)) # 水平合并
print(ae.shape,ae2.shape,ae3.shape) # 3个元素的序列，2行3列，6个元素的序列
ae4 = ae[np.newaxis,:] # np.newaxis可以显示增加了1个维度
ae5 = ae[:,np.newaxis] # 在ae的array里多添加一个[]也可以显示行列数
print(ae4)
print(ae5)
print(ae4.shape,"\n",ae5.shape)
ae6 = np.vstack((ae,ae1,ae1,ae))
ae7 = np.concatenate(([[1,2],[23,2]],[[1,45],[23,35]]),axis =0) # axis=0表示竖直合并，1表示水平合并
print(ae6,"\n",ae7)

# 分割
af = np.arange(12).reshape(3,4)
af1 = np.split(af,3,axis = 0) # 3表示分成3块；axis=0表示横向分割，axis=1代表纵向分割
print(np.array_split(af,3,axis = 1))
print(af1)
print(np.vsplit(af,3)) # vsplit为竖直（添加）分割
print(np.hsplit(af,2)) # hsplit为水平（添加）分割

# 赋值
ag = [1,2,3,4]
bg = ag
cg = bg
print(ag,bg,cg)
bg[1:3] = [22,33]
print(ag,bg,cg)
dg = ag.copy() # deep copy dg不随ag改变
print(dg,dg is ag)

### pandas
s = pd.Series([1,3,6,np.nan,4,1])
print(s) # 序列
dates = pd.date_range('20230708',periods = 6)
print(dates) # 日期
df = pd.DataFrame(np.random.randn(6,4),index = dates,columns = ['a','b','c','d'])
print(df) # 定义行列和内容
print()
df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)
print(df.dtypes)
print(df.columns)
print(df.values)
print(df.T) # 矩阵转置
print(df.sort_index(axis = 1,ascending= False)) # axis=1为列标，ascending=False将行或列以倒序排序
print(df.sort_index(axis = 1,ascending= True))
print(df.sort_values(by='b'))

# 数据筛选
print(df['a'],df.a)
print(df.loc['20230708'])
print(df.loc[:,['a','b']]) # 打印出a列和b列
print(df.iloc[[1,3,5],1:3])# 打印出第一行、第三行和第五行，第一列到第三列
print(df[df.c>1])

# 设置值
df.b[df.a>1] = 0 # 将a中大于1的值所在行中的b的值改为0
print(df)
df.iloc[2,2] = 111 # 将第二行第二列的值改为111
print(df)
df.loc['20230708','b'] = 12 # 将行为'20230708',列为'b'的值改为12
print(df)
df[df.c>1] = 0 # 将c中大于1的值所在行中的所有值都改为0
print(df)
df['e'] = np.nan # 添加'e'这一列
print(df)
df['f'] = pd.Series([1,2,3,4,5,6],index = pd.date_range('20230708',periods = 6))
print(df)
df.iloc[0,4] = 1 # 将第0行第四列的数据改为1
print(df)

# 处理丢失数据
print(df.dropna(axis = 0,how = 'any')) # 丢掉含有nan的所有行 how = {'any','all'}
print(df.fillna(value = 0)) # 将所有的nan改为0
print(df.isnull()) # true表示是nan，false表示不是nan
print(np.any(df.isnull()) == True) # 判断是否有nan，true表示有

# 导入导出
data = pd.read_csv('test_1.csv',encoding='gbk')
print(data)
# data.to_pickle('student.pickle')
data.iloc[0,1] = 'ccc'
print(data)

# 合并concatenating
df1 = pd.DataFrame(np.zeros((3,4)),columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4)),columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2,columns = ['a','b','c','d'])
print(df1,df2,df3)

# ignore_index()行号从零开始排序
res = pd.concat([df1,df2,df3],axis = 0,ignore_index=True) # 纵向合并；ignore_index = True可以将行索引从零开始依次排序
print(res)

# join()
df4 = pd.DataFrame(np.ones((3,4))*0,index = [1,2,3],columns=['a','b','c','d'])
df5 = pd.DataFrame(np.ones((3,4)),index = [2,3,4],columns=['b','c','d','e'])
print(df4,df5)

res1 = pd.concat([df4,df5] , join = "inner" , ignore_index= True) # join = 'inner'，表示把共有的留下；join = 'outer'是默认的，表示合并后把原来没有的用nan显示
print(res1)
res2  = pd.concat([df4,df5],axis= 1) # 横向和并
res2_ = pd.concat([df4,df5.reindex(df4.index)],axis= 1) # 按照df4的行将df4与df5横向合并
print(res2,'\n',res2_)

# append() 上下合并
df6 = pd.DataFrame(np.ones((3,4))*0,columns= ['a','b','c','d'])
df7 = pd.DataFrame(np.ones((3,4)),columns=['a','b','c','d'])
df8 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
res3 = df6.append([df7,df8],ignore_index=True)
print(res3)
s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
res4 = df6.append(s1,ignore_index=True)
print(res4)

# 合并 merge
left = pd.DataFrame({'Key':['k0','k1','k2','k3'],'A':['a0','a1','a2','a3'],'B':['b0','b1','b2','b3'],'Key1':['k0','k1','k0','k2']})
right = pd.DataFrame({'Key':['k0','k1','k2','k3'],'C':['c0','c1','c2','c3'],'D':['d0','d1','d2','d3'],'Key1':['k1','k1','k0','k1']})
print(left,right)
res5 = pd.merge(left,right,on = 'Key') # on = '以共有的列名称合并'
print(res5)
res6 = pd.merge(left,right,on = ['Key','Key1'],how = 'left') # how =['outer','inner','left','right'] 'outer'表示把key和key1相对应的值都合并起来，'inner'表示把二者共有的合并起来，'left'表示只合并左边，'right'表示只和并右边
print(res6)
df9 = pd.DataFrame({'cool':[0,1],'coo':['a','b']})
df10 = pd.DataFrame({'cool':[1,2,3],'co':[1,3,5]})
res7 = pd.merge(df9,df10,on = 'cool',how = 'outer',indicator=True) # indicator = True显示合并的方式 仅左边，仅右边，都有
print(res7)
left1 = pd.DataFrame({'a':['a0','a1','a2'],'b':['b0','b1','b2']},index = ['k0','k1','k2'])
right1 = pd.DataFrame({'c':['c0','c1','c2'],'d':['d0','d1','d2']},index = ['k0','k2','k3'])
res8 = pd.merge(left1,right1,left_index=True,right_index=True,how='outer') # 令左右矩阵都以行名称合并
print(res8)
boys = pd.DataFrame({'k':['k0','k1','k2'],'age':[1,2,3]})
girls = pd.DataFrame({'k':['k0','k0','k3'],'age':[6,7,8]})
res9 = pd.merge(boys,girls,on = 'k',suffixes=['_boy','_girl'],how = 'outer') # suffixes将共有的age这列划分为两列（age_boy,age_girl）
print(res9)


### matplotlib
### ctrl + b 或者 ctrl + 鼠标点击代码 可以查看源码
import matplotlib.pyplot as plt 
# plt.plot() 绘制出图形
# plt.show() 展示图形

# x = range(1,10,2)
# y = [1,5,7,5,2]
# # plt.figure(figsize=(2,2),dpi = 300)
# plt.plot(x,y)
# x_labels = [0,1,2,3,4,5,6,7,8,9,10]
# plt.xticks(x_labels[::2]) # 定义x轴的刻度 range(0,11)； [::2]步长为2
# plt.yticks(range(min(y),max(y)+1))
# plt.show()
#  # 保存 plt.savefig('./test1.png')

x1 = list(range(120))
y1 = [random.randint(1,100) for i in range(120)]
plt.plot(x1,y1)
x1_labels = [ '10点{}分'.format(i) for i in range(60)]
x1_labels +=[ '11点{}分'.format(i) for i in range(60)]
plt.xticks(x1[::20],x1_labels[::20],rotation = 0) # rotation = 45 旋转45度
plt.xticks(fontproperties = 'STSong') # 让横轴标签显示中文
plt.xlabel('时间',fontproperties = 'STSong')
plt.ylabel('温度 单位（℃）',fontproperties = 'STSong',rotation =  0)
plt.title('10点到12点每分钟的气温变化情况',fontproperties = 'STSong')
plt.show()
plt.close()

import matplotlib.pyplot as plt
x2 = list(range(10,21))
y2 = [0,1,2,5,8,5,3,10,24,6,12]
y22 = [1,2,3,6,17,3,7,2,7,19,20]
plt.plot(x2,y2,label='同学1',color = 'blue',linestyle = '-.',linewidth = 2) # 图例中显示 label是线的名字，颜色是蓝色，线性是点画线,线宽是5
plt.plot(x2,y22,label='同学2',color = 'cyan',linestyle = '--')
plt.rcParams['font.family'] = ['Microsoft YaHei'] # 使用中文字体
x2_label = ['{}岁'.format(i) for i in range(10,21)]
plt.xticks(x2,x2_label,rotation = 0,fontproperties = 'STSong')
plt.yticks(list(range(0,31))[::3])
plt.xlabel('年龄',fontproperties = 'STSong')
plt.ylabel('成就数',fontproperties = 'STSong')
plt.title('成长历程',fontproperties = 'STSong')
plt.grid(alpha = 0.3,linestyle = ':')
plt.legend(loc = 'upper left')
plt.show()
plt.close()

# 绘制散点图 plt.scatter()
import matplotlib.pyplot as plt
x3_1 = range(1,32)
x3_2 = range(50,80)
y3_1 = [random.randint(1,30) for i in range(31)]
y3_2 = [random.randint(1,45) for i in range(30)]
plt.scatter(x3_1,y3_1,label = '洛阳',color = 'b')
plt.scatter(x3_2,y3_2,label = '天津',color = 'g')
x3_label = ['1月{}号'.format(i) for i in range(31)]
x3_label += ['9月{}号'.format(i) for i in range(30)]
x3_ = list(x3_1) + list(x3_2)
plt.rcParams['font.family'] = ['Microsoft YaHei'] # 使用中文字体
plt.xticks(x3_[::5],x3_label[::5],fontproperties = 'STSong',rotation = 45)
plt.xlabel('时间',fontproperties = 'STSong')
plt.ylabel('温度（单位：℃）',fontproperties = 'STSong')
plt.title('两地温度图',fontproperties = 'STSong')
plt.legend()
plt.show()
plt.close()

# 绘制条形图 plt.bar()
import matplotlib.pyplot as plt
a = ['movie1','movie2','movie3','movie4','movie5','movie6','movie7','movie8','movie9','movie10']
b = [11,24,32,61,57,43,23,12,38,27]
plt.xticks(range(10),a,fontproperties = 'STSong')
plt.xlabel('电影名称',fontproperties = 'STSong')
plt.yticks(range(0,max(b)+10,5))
plt.ylabel('票房（单位：亿）',fontproperties = 'STSong')
plt.title('电影票房统计')
plt.bar(a,b,width= 0.8,color = 'orange')
plt.grid(alpha = 0.3)
plt.show()
plt.close()
plt.barh(a,b,height = 0.5, color = 'r')
plt.xticks(range(0,max(b)+10,5))
plt.xlabel('票房（单位：亿）',fontproperties = 'STSong')
plt.title('电影票房统计')
plt.grid(alpha = 0.3)
plt.show()
plt.close()

import matplotlib.pyplot as plt
b1 = [257,451,527,692]
b2 = [414,578,579,789]
b3 = [526,743,632,467]
b4 = [356,235,346,438]
x = list(range(len(a)))
x_1 = list(range(1,40,11))
x_2 = [i + 2 for i in x_1]
x_3 = [i + 2*2 for i in x_1]
x_4 = [i + 2*3 for i in x_1]
plt.yticks(range(0,801,100))
plt.ylabel('票房（单位：亿）',fontproperties = 'STSong')
x_ = ['1月{}日'.format(i) for i in range(12,16)]
plt.xticks(range(4,40,11),x_,fontproperties = 'STSong')
plt.xlabel('时间')
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.bar(x_1,b1,label = 'movie1',color = 'r',width = 1.4)
plt.bar(x_2,b2,label = 'movie2',color = 'g',width = 1.4)
plt.bar(x_3,b3,label = 'movie3',color = 'b',width = 1.4)
plt.bar(x_4,b4,label = 'movie4',color = 'c',width = 1.4)
plt.grid(alpha = 0.3)
plt.legend()
plt.title('1月12日-1月16日票房统计')
plt.show()
plt.close()

# 绘制直方图 
# plt.hist(a,num_bins,normed) bin_width为组距 num_bins为组数 num_bins = int((max(aa) - min(aa))/bin_width)
aa = [73,23,84,28,61,62,61,63,94,33,44,55,66,77,88,99,11,22,1,3,5,7,25,68,27,95,66,89,57,27,54,36,16,48,16,98,26,56,25,43,56,32]
bin_width = 5
num_bins = int((max(aa) - min(aa))/bin_width)
plt.hist(aa,range(min(aa),max(aa)+bin_width,bin_width),density=0)
plt.xticks(range(min(aa),max(aa)+bin_width,bin_width))
plt.xlabel('数据值', fontproperties = 'STSong')
plt.ylabel('个数', fontproperties = 'STSong')
plt.grid(alpha = 0.3,linestyle = ':')
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.title('直方图')
plt.show()
plt.close()
# 组数不均匀
# num_bins_uneven = [min(aa)*i*bin_width for i in range(num_bins)]
# plt.hist(aa,num_bins_uneven)
# plt.show()

import matplotlib.pyplot as plt
interval = [0,5,10,15,20,25,30,35,40,45,60,90]
width_ = [5,5,5,5,5,5,5,5,5,15,30,60]
quantity = [1223,2441,3521,4251,2355,6642,5754,4527,6782,8782,3727,6583]
plt.xlabel('时间间隔（单位：分钟）',fontproperties = 'STSong')
plt.ylabel('人数',fontproperties = 'STSong')
x__ = ['{}分钟'.format(i) for i in interval]
plt.xticks(list(range(len(interval)))[::2],x__[::2],fontproperties = 'STSong')
y_ = range(min(quantity),max(quantity)+1000,500)
plt.yticks(range(1000,10000,500))
plt.bar(range(len(quantity)),quantity,width = 1)
plt.grid(alpha = 0.3, linestyle = ':')
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.title('直方图')
plt.show()
plt.close()

data = pd.Series(np.random.randn(1000),index = np.arange(1000))
print(data)
# data = data.cumsum()
# data.plot()
# plt.show()

data1 = pd.DataFrame(np.random.randn(1000,4),
                     index = np.arange(1000),
                     columns= list('ABCD'))
print(data1)
# data1 = data1.cumsum()
# data1.plot()
# plt.show()

#plot methods-- bar hist pie area scatter box kde hexbin...
pp = data1.plot.scatter(x = 'A',y = 'B',color='Red',label = 'Class 1')
data1.plot.scatter(x = 'A',y = 'C',color = 'Blue',label = 'Class 2',ax = pp)
