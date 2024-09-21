#for循环

for i in range(1,11):
    print("第",i,"次循环，当前的值是",i)
print("循环结束")

#累加
sum = 0
for i in range(1,101):
    sum += i
    print(sum)
print("1到100的累加结果为",sum)

numbers = [1,2,3,4,5]
sum = 0
for i in numbers:
    print(i)
    sum += i
print("1到5的累加结果为",sum)

#累乘
a = 1
for i in range(1,11):
    a *= i
    print(a)
print("1到10的累乘结果为",a)

#遍历
b = 0
for i in "Hello,world!":
    if i != "!":
        b += 1
        print(b)
        print(i)
    else:
        print(i)
print(b)

#双重循环
a = [1,2,3]
b = [1,2,3]
for i in a:
    for j in b:
        print([i*j])
#九九乘法表
q = [1,2,3,4,5,6,7,8,9]
w = [1,2,3,4,5,6,7,8,9]
for i in q:
    for j in w:
        if i >= j:
            print("{:2}*{:2}={:2}".format(i,j,i*j),end=",")
    print()

#break continue
numbers_ = [1,2,3,4,5,6,7,8,9,10]
for i in range(1,11):
    if i == 6:
        print("找到老6了")
        break
    else:
        print(i)
for i in numbers_:
    if i % 2 == 0 : #找所有的奇数
        continue
    print(i)

#for else
for i in "中国民航大学！":
    if i == "！":
        break
    print(i,end="")
else:
    print("正常退出")

print()
print("aa">"a1")
print("abdcefg"[1:-1:2])
print(chr(48),chr(65),chr(97),ord("0"))
print("abdc efgh".split())
print("I LOVE YOU".center(26,"-"))
print("asdfgh".replace("asd","111"))
print("我爱你".strip("我"))
print("，".join("我想你了"))
