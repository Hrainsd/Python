#第一题
time = 0
result = []
for i in range(1000,10000):
    zfc = str(i)
    zs= [int(zfc[i]) for i in range(4)]
    he = sum(zs)
    ys = i % 13
    a = zs[0]-zs[1]
    b = zs[0]-zs[2]
    c = zs[0]-zs[3]
    d = zs[1]-zs[2]
    e = zs[1]-zs[3]
    f = zs[2]-zs[3]
    if a!=0 and b!=0 and c!=0 and d!=0 and e!=0 and f!=0 and he == 8 and ys == 0:
            time += 1
            result.append(i)
print(time)
print(result)

#第二题
import random
my_list = []
result = []
n = int(input("请输入n的值："))
for i in range(n):
    random.seed()
    a = random.randint(0,9)
    my_list.append(str(a))
print(my_list)
my_set = set(my_list)
result = list(my_set)
result.sort()
print(result)

#第三题
lst1 = [1,2,3,5,6,3,2]
lst2 = [2,5,7,9]
lst1set = set(lst1)
lst2set = set(lst2)
jj = lst1set & lst2set
cj = lst1set - jj
bj = lst1set|lst2set
print("交集为：",jj)
print("并集为",bj)
print("差集为：",cj)
