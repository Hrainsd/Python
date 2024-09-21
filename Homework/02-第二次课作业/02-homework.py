#第一题第一问
a = float(input("请输入正方形的边长："))
s_1 = a*a #s_1表示正方形的面积
v_1 = a*a*a #v_1表示正方体的体积
print("正方形的面积为",str(s_1))
print("正方体的体积为",str(v_1))

#第一题第二问
r = float(input("请输入半径的值："))
pi = 3.142
s_2 = pi*(r**2) #s_2表示圆的面积
v_2 = 4*pi*(r**3)/3 #v_2表示球的体积
print("圆的面积为",str(s_2))
print("球的体积为",str(v_2))

#第二题
b = float(input("请输入数值："))
if b > 0 :
    print("您输入的数为正数")
elif b == 0 :
    print("您输入的数为0")
else :
    print("您输入的数为负数")

#第三题
Year = int(input("请输入年份："))
if (Year%4==0 and Year%100!=0)or(Year%400==0):
    print(str(Year)+"年是闰年！")
else:
    print(str(Year)+"年不是闰年！")

#第四题
Year = int(input("请输入年份："))
second_1 = 366*24*60*60
second_2 = 365*24*60*60
if (Year%4==0 and Year%100!=0)or(Year%400==0):
    print(str(Year)+"年有"+str(second_1)+"秒")
else:
    print(str(Year)+"年有"+str(second_2)+"秒")
