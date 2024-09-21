a = [1,2,3,4,5,6,7,8,9]
b = a[3:5]
print(b)
c = str(a)
print(c)
print(type(c[2]))
d = c.strip("[]").split(",")
print(d)
print("".join(d))

#第一题
s = str(input("请输入您想要反向输出的字符串："))
print(s[::-1])

#第二题
name = str(input("请输入您的姓名："))
if "王" in name:
    print("您的姓是”王“")
else:
    print("您的姓不是”王“")

#第三题
a = str(input("请输入一个字符串："))
print(a[::2])
