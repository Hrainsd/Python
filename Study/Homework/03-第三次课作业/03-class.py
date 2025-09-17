#if语句

if 2 > 1 :
    print("2大于1")
    print("这句话会被执行")

age = int(input("请输入您的年龄："))
if age > 18:
    if age > 25:
        if age > 45:
            if age > 60:
                print("您已步入老年阶段")
            else:
                print("您已步入中年阶段")
        else:
            print("您已步入壮年阶段")
    else:
        print("您已步入青年阶段")
else:
    if age > 15:
        print("您已步入青少年阶段")
    else:
        print("您已步入童年阶段")
print(divmod(55,10))#divmod(x,y)得结果是(x//y,x%y)
