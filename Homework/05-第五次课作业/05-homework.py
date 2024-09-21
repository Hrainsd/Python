#水仙花数
for i in range(100,1000):
    a = str(i)
    b = int(a[0])
    c = int(a[1])
    d = int(a[2])
    if b**3 + c**3 + d**3 == i:
        print(i)

#第一题
a = float(input('a='))
b = float(input('b='))
c = float(input('c='))
max_ = max(a,b,c)
print("最大的数为："+str(max_))

#第二题
s = str(input("请输入字符串："))
for i in s:
    print(i, end=",")
print()
for i in s[::-1]:
    print(i, end=",")
print()
#第三题第一种方法
f = input("请输入您想求和的各个数（数与数之间用英文逗号隔开）：")
f = f.split(",")
total = sum(map(float, f))
print(total)
#第三题第二种方法
h = input("请输入您想求和的各个数（数与数之间用英文逗号隔开）：")
h = h.split(",")
total = 0
for i in h:
    total +=float(i)
print(total)
