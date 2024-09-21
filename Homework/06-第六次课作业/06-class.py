#序列

fruits = ["苹果","香蕉","葡萄","黄桃"]
print(fruits[0])
fruits.append("桑葚") #添加一个元素到列表的最后
for fruit in fruits:
    print(fruit)

ab = ["人","善","狗","欺"]
ab[0:2] = ["人不能","太善良"] #用等号后面的代替列表从0开始到1的元素
print(ab)
del ab[0:2] #删除列表从0开始到1的元素
print(ab)
print(ab*2) #重复列表里的元素
ab.append("远离垃圾人")
print(ab)
ab.insert(0,"人善") #0的位置插入元素
print(ab)
ab.reverse() #反序列表里的元素
print(ab)
ab.index("狗") #"狗"所在的位置
print(ab.index("狗"))

dd = [1,2,3,4,5]
dd.sort() #默认reverse = False 也就是升序
print(dd)
dd.sort(reverse=True)
print(dd)

import random
e = random.randint(1,10)
print("你选出的数字为",e)

#成绩统计
import random
grades = []
for i in range(10):
    grade = random.randint(0,100)
    grades.append(grade)
print(grades)
average = sum(grades)/len(grades)
print("平均分为：",average)
highest = max(grades)
lowest = min(grades)
print("最高分为",highest,",最低分为",lowest)
grades.sort(reverse=True)
print(grades)

#zip函数 lambda函数
names = ["艾伦","三笠"]
scores = [21,521]
students = list(zip(names,scores))
_unname = sorted(students, key=lambda x:x[1], reverse=True)
print(_unname)

add = lambda x,y,z:(x**2+y**2+z**2)**(1/2)
f = add(6,6,6)
print(f)
