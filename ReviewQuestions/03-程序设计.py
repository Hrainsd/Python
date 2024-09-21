"""第一题
number +=1
total +=chengji
while True:
"""

"""第二题
import turtle
for i in range(5):
    turtle.fd(100)
    turtle.left(72)
"""

"""第三题
pi = 0
i = 1
n = 0
while 1/i > 0.01:
    pi += (-1)**n/i
    i += 2
    n += 1
print("{:.6f}".format(4*pi))
"""

"""第四题
a = float(input())
y = "0."
while True:
    b = str(a*2)
    y += b[0]
    c = "0."+b[2:]
    a = float(c)
    if int(b[2:]) == 0 or len(y) == 18:
        print(y)
        break
"""

"""第五题
import random
a = "ABCDEFGHIJ0123456789"
n = int(input())
random.seed(n)
for i in range(6):
    m = random.randint(0,19)
    print(a[m],end="")
"""

"""第六题
import random
a = random.randint(1,100)
while True:
    b = int(input())
    if b not in [i for i in range(1, 101)]:
        print("输入数据不在1~100之间")
        break
    elif a == b :
        print("恭喜您猜对了")
        break
    elif a < b:
        print("您的猜测太大了")
    elif a > b:
        print("您的猜测太小了")
"""

"""第七题
a = input()
b = ""
for i in a:
    if i not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
        b += i
    elif 64 < ord(i) <91:
        c =  str(chr(155-ord(i)))
        b += c
    elif 96 < ord(i) < 123:
        d = str(chr(219-ord(i)))
        b += d
print(b)
"""

"""第八题
a = input()
b = list(a)
c = []
for i in b:
    c.insert(0,i)
d = "".join(c)
print(int(d))
"""

"""第九题
months = "Jan.Feb.Mar.Apr.May.Jun.Jul.Aug.Sep.Oct.Nov.Dec."
a = input()
n = int(a)
if a in [str(i) for i in range(1,13)]:
    print(months[(n-1)*4:4*n])
else:
    print("请输入1~12之间的数字")
"""

"""第十题
a = input().split(",")
b = list(set(a))
c = sorted(b,key=a.index)
print(c)
"""

"""第十一题
a = input().upper()
b = len(a)
if b == 2 :
    if a[0] in "123456789" and a[1] in "ABCDF":
        if a[1] in "AF" :
            print("窗口")
        elif a[1] in "CD" :
            print("过道")
        else:
            print("中间")
    else:
        print("座位号不存在")
elif b == 3 :
    if a[:2] in [str(i) for i in range(10,18)] and a[2] in "ABCDF":
        if a[2] in "AF" :
            print("窗口")
        elif a[2] in "CD" :
            print("过道")
        else:
            print("中间")
    else:
        print("座位号不存在")
else:
    print("座位号不存在")
"""

"""第十二题
a = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
b = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26"]
c = {}
for i in range(26):
    c[a[i]] = b[i]
d =input().upper()
e = ""
for j in d:
    if j in c:
        e += c[j]
    else:
        e += j
print(e)
"""

"""第十三题
a = {"河南":"郑州","江苏":"南京","浙江":"杭州"}
while True:
    pro = input()
    if pro == "0":
        break
    else:
        print(a.get(pro,"输入错误"))
"""

"""第十四题
a = int(input())
dic = {"张三":["1","a"],"王五":["2","b"],"赵四":["3","c"]}
if a == 4:
    b = input()
    if b in dic :
        print(b,dic[b][0],dic[b][1])
        print("Success")
    else:
        print("No Record")
else:
    print("ERROR")
"""

"""第十五题
dic = {"张三":"1","王五":"2","赵四":"3"}
i = 1
while True:
    a = input() #用户名
    b = input() #密码
    if a in dic and b == dic[a] :
        print("登陆成功")
        break
    else:
        if i < 3 :
            print("登陆失败")
            i += 1
        else:
            print("登陆失败")
            break
"""

"""第十六题
res = []
def isPrime(n):
    for i in range(2,n+1):
        a = 0
        for j in range(1,i+1):
            if i%j == 0:
                a += 1
        if a == 2:
            res.append(i)
    for k in res:
        print(k,end=" ")
n = int(input())
isPrime(n)
"""

"""第十七题
def fun(m,n):
    s = m
    while True:
        p = s%m
        q = s%n
        if p == 0 and q == 0:
            zxgbs = s
            break
        else:
            s += 1
    while True:
        c = m%n
        if c == 0:
            zdgys = n
            break
        else:
            m = n
            n = c
    return (zxgbs,zdgys)
def main():
    a = int(input("请输入一个整数："))
    b = int(input("请输入另一个整数："))
    print("[",a,"和",b,"]的最小公倍数和最大公约数为：",fun(a,b))
if __name__ == '__main__':
    main()
"""

"""第十八题
def main():
    with open("test1.txt","r") as a:
        s1 = a.read()
    with open("test2.txt","r") as b:
        s2 = b.read()
    s3 = list(s1 + s2)
    s4 = sorted(s3)
    s = "".join(s4)
    with open("test3.txt","w") as c:
        c.write(s)
if __name__ == '__main__':
    main()
"""
"""a = "i like apple"
a.replace("like","love")
print(a) #.replace产生一个副本

a = (1,2,3)*2
print(a)



Q = {"a":1,"b":2}
print(Q["a"])
while True:
    t1 = float(input())
    t2 = float(input())
    a = 4*3.14*(2/t2-1/t1)/(t2-t1)
    print(a)


a = input().upper()
n = list(a)
b = len(a)
if b not in [2,3] :
    print("座位号不存在")
elif b == 2:
    if n[0] in "123456789" and n[1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if n[1] == "A" or "F":
            print("窗口")
        elif n[1] == "C" or "D":
            print("过道")
        elif n[1] == "B":
            print("中间")
    else:
        print("座位号不存在")
elif b == 3:
    if int(a[:2]) in range(1,18) and n[2] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if n[2] == "A" or "F":
            print("窗口")
        elif n[2] == "C" or "D":
            print("过道")
        elif n[2] == "B":
            print("中间")
    else:
        print("座位号不存在")









a = list(input())
c = len(a)
z = []
for i in a:
    if i in "0123456789":
        z.append(i)
    else:
        if ord(i) <= 90:
            o = 90 - (ord(i) - 65)
            z.append(chr(o))
        else:
            o = 122 - (ord(i) - 97)
            z.append(chr(o))
print("".join(z))

pi = 0
n = 0
i = 1
while 1/i > 0.01:
    pi += (-1)**n/i
    n += 1
    i +=2
print("{:.6f}".format(4*pi))

a = float(input("输入一个0-1之间的小数："))
b = "0."
while True:
    c = 2*a
    b += str(c)[0]
    d = "0."+str(c)[2:]
    a = float(d)
    if float(str(c)[2:]) == 0 or len(b[2:]) == 16:
        print(b)
        break

import random
f = "ABCDEFGHIJ0123456789"
g = len(f)
r = ""
n = int(input())
random.seed(n)
for i in range(6):
    index = random.randint(0,g-1)
    r += f[index]
print(r)"""

# #一个目录就是一个包 一个文件就是一个模块
# import sys #不同路径下的引用
# import os
# pa = r"C:\Users\23991\OneDrive\桌面\Python\venv\share" #去掉最后一个C:\Users\23991\OneDrive\桌面\Python\venv\share\kua_mu_lu.py
# sys.path.append(pa)
# print(sys.path) #sys.path 是模块查找路径
# import kua_mu_lu
# kua_mu_lu.ddd()
# print(__file__)
