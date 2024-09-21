#第一题
shuru = input("请输入十个整数：")
sr =  shuru.split(" ")
lb = [int(sr[i])for i in range(10)]
a = 0
for i in range(10):
    for j in range(i+1,10):
        if (lb[i]) < (lb[j]):
            lb[i],lb[j] = lb[j],lb[i]
for m in range(0,10):
    if lb[m]%2 == 1:
        a += 1
        print("最大的奇数为："+str(lb[m]))
        break
if (a==0):
    print("您没有输入奇数。")

#第二题
password = 666
i = 0
while i < 3 :
    inp = int(input("请输入用户密码："))
    if password == inp :
        print("恭喜您，输入正确。")
        break
    else:
        i += 1
        if i < 3:
            print("密码错误，您还有"+str(3-i)+"次机会。")
        else:
            print("您已输入错误三次，不可再输入。")

#第三题
import random
num = random.randint(1,100)
count = 0
while count < 6:
    guess = int(input("请猜一下数字是多少？(1-100之间)："))
    if guess == num:
        print("恭喜你猜对了！")
        break
    elif guess < num:
        print("你猜的数字小哦。")
    else:
        print("你猜的数字大哦。")
else:
    print("很遗憾，你没有猜对。")
