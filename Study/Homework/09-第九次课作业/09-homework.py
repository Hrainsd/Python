#第一题
def jiecheng(n):
    result = 1
    for i in range(1,n+1):
        result *= i
    return result
n = eval(input("请输入n的值："))
jc_ = jiecheng(n)
print(jc_)
a_1 = jiecheng(1)
a_2 = jiecheng(2)
a_3 = jiecheng(3)
a_4 = jiecheng(4)
a_5 = jiecheng(5)
a_6 = jiecheng(6)
a_7 = jiecheng(7)
a_8 = jiecheng(8)
a_9 = jiecheng(9)
a_10 = jiecheng(10)
sum = a_1+a_2+a_3+a_4+a_5+a_6+a_7+a_8+a_9+a_10
print(sum)

#第二题
def three(*numbers):
    total = 0
    max_ = max(numbers)
    min_ = min(numbers)
    for num in numbers:
        total += num
    return max_,min_,total/len(numbers)
numbers_ = input("请输入任意多个数（数与数用空格隔开）：").split()
lb = list(map(float, numbers_))
a,b,c = three(*lb)
print("最大值为：",a,"最小值为：",b,"平均值为：",c)

#第三题
def inf(list):
    for i in list:
        if list[0] == "Q":
            exit()
        elif list[0] == "q":
            exit()
        elif list[1] == "女":
            return list[0],list[1],list[2],list[3]
        else:
            list[1] = "男"
            return list[0],list[1],list[2],list[3]
inf_ = input("请输入您的姓名，性别，年龄和学历（信息之间用空格隔开）：").split()
a,b,c,d = inf(inf_)
print("姓名：",a,"性别：",b,"年龄：",c,"学历：",d)
