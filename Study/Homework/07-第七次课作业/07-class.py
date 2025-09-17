#while
i = 1
sum = 0
while i <= 10:
    print("i=",i)
    sum += i
    i += 1
else:
    print("sum=",sum)

#while循环猜数
import random
num = random.randint(1,10)
count = 0
while count < 5:
    guess = int(input("请猜一下数字是多少？(1-10之间)："))
    if guess == num:
        print("恭喜你猜对了！")
        break
    elif guess < num:
        print("你猜的数字小哦。")
    else:
        print("你猜的数字大哦。")
else:
    print("很遗憾，你没有猜对。")

#水仙花数
sxh = []
for num in range(100,1000):
    a = num//100
    b = (num//10)%10
    c = num%10
    if a**3+b**3+c**3 == num:
        sxh.append(num)
print(sxh)

for num in range(100,1000):
    s = str(num)
    if int(s[0])**3+int(s[1])**3+int(s[2])**3 == num:
        print(num,end=" ")

n = "中国民航大学 空管学院 交通管理"
n_ = n.split(" ")
for i in n_:
    print(i)
n__ = "，".join(n_)
print(n__)

#爱心
for i in range(6):
    for j in range(7):
        if (i==0 and j%3!=0) or (i==1 and j%3==0) or (i-j==2) or (i+j==8):
            print("*",end="")
        else:
            print(" ",end="")
    print()
