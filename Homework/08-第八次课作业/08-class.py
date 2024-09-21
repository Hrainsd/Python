#字典
fruit_price = {"芒果":3,"葡萄":5,"西瓜":6}
print("芒果的价格为",fruit_price["芒果"],"元。")
fruit_price["桃"] = 4 #添加键值对："桃":4
print(fruit_price)
for key in fruit_price: #字典的遍历
    print(fruit_price[key])
for key,value in fruit_price.items(): #items方法的遍历
    print(key,value)
for key in fruit_price.keys(): #keys方法的遍历
    print(key)
for value in fruit_price.values(): #values方法的遍历
    print(value)
#成绩统计
grades = {"小枫":96,"小舟":98,"小兰":66}
total = 0
for name,grade in grades.items():
    total += grade
average = total/len(grades)
print("班级平均分为：",average)

grade_ = {"小玉":{"语文":88,"数学":98,"英语":100},"小艾":{"语文":66,"数学":78,"英语":90}}
for i in grade_:
    total_ = grade_[i]["语文"]+grade_[i]["数学"]+grade_[i]["英语"]
    print(i+"的总成绩是："+str(total_)+"分。")

#set函数
colors = set(["红色","红色","橙色","蓝色","紫色"])
print(colors)
co = set("cyh11616")
print(co)


#交集，并集，差集，补集
a = {1,3,5}
b = {2,3,4}
c = a&b
d = a|b
e = a-b
f = a^b
print(c,d,e,f)

#元组（不能改表）
tuple_ =(6666,"你真牛啊")
print(tuple_[0])
creature = "human","pig","dog"
print(creature)
creature_ = ("god",creature,"dirt")
print(creature_)
