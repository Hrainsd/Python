#第一题
a = int(input("请输入一个整数："))
if a%2 == 0:
    print(str(a)+"是一个偶数")
else:
    print(str(a)+"是一个奇数")

#第二题
b = int(input("请输入一个三位数："))
b1 = b//100 #百位上的数字
b2 = (b%100)//10 #十位上的数字
b3 = b%10 #个位上的数字
c = b1 + b2 + b3
print("该三位数的百位数字为："+str(b1))
print("该三位数的十位数字为："+str(b2))
print("该三位数的个位数字为："+str(b3))
print("该三位数的各个位的数字之和为"+str(c))
#第二题拓展
for d in range(100,1000):
    x = d//100
    y = (d%100)//10
    z = d%10
    e = (x)**3+(y)**3+(z)**3
    if d == e:
        print(d)

#第三题
grade = int(input("请输入一个测验成绩（0—100）："))
if grade >59:
    if grade >69:
        if grade >79:
            if grade >89:
                print("该成绩的评分为：A")
            else:
                print("该成绩的评分为：B")
        else:
            print("该成绩的评分为C")
    else:
        print("该成绩的评分为D")
else:
    print("该成绩的评分为：F")
