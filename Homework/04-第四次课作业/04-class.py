se = input("请输入您的座位号：")
seat = str(se)
ck = ["a","A","f","F"]
if seat[1] in ck  :
    print("chenggong")
    int(seat[0]) ==1 and int(seat[1]) in range(10)
elif int(seat[0]) == 2 and int(seat[1]) in range(9):
    if seat[2] in ck:
        print("您的位置在窗口")
elif seat[2] in zj:
    print("您的位置在中间")
elif seat[2] in gd:
    print("您的位置在过道")
else:
    print("座位号不存在")

"""字符串
相关知识"""
'''字符串的正序从0开始
反序从-1开始'''
a = "apple"
print(a[0])
print(a[4])
print(a[-1])
print(a[-5])
b = a[0]
print(b)

#字符串遍历
c = "中国民航大学"
for s in c :
    print(s)

#字符串切片   字符串[m:n:k]从m开始到n之前结束，不含n，k表示步长，步长为负表示从右往左取。
d = a[0:3]
print(d)
print(a[0:4])
print(a[0:5:2])
print(a[::-1]) #没有m没有n表示从头开始到尾结束

#字符串操作符 (+ * in)
print(a+c)
print(a*8)
if b in a :
    print(b+"包含在"+a+"里面。")
else :
    print("不包含")

#字符串处理函数 len str hex整数的十六进制或者八进制小写形式字符串 chr返回对应的字符 ord返回对应的unicode编码
print(len(c))
print(ord("a"))
print(chr(97))

#字符串处理方法 str.lower() str.upper() str.split() str.count() str.replace(old,new) str.center(width,"符号")
#str.strip() str.join() str.format()
e = "A BCde f"
print(e.lower()) #全部小写
print(e.upper()) #全部大写
print(e.split()) #以空格分离
print(e.count("A")) #"A"出现的次数
print(e.replace("A BC","abc")) #用逗号后面的代替逗号前面的
print(e.center(20)) #在长度为width的字符串中居中显示
print(e.strip("Af")) #在e中去掉()里列出的字符
print(",".join(e)) #在e的每个元素后增加一个逗号，除了最后一个元素
f = "中国民航大学"
g = "校长"
h = "丁水汀"
print("{}的{}是{}".format(f,g,h))
