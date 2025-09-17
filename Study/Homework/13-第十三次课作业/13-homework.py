#第一题
for i in range(1,10):
    for j in range(1,10):
        result = i * j
        print(i,"×",j,"=",result,end="  ")
    print()

#第二题
n = int(input("请输入整数n："))
total = 0
lb = []
for i in range(1,n+1):
    total += int(i*"1")
    lb.append(int(i*"1"))
print(lb)
print("前n项的和为",total)

#第三题
se = input("请输入您的座位号：")
seat = str(se)
ck = ["a","A","f","F"]
zj = ["b","B","e","E"]
gd = ["c","C","d","D"]
if len(seat) == 2:
    if int(seat[0]) in range(1,10):
        if seat[1] in ck:
            print("您的位置在窗口")
        elif seat[1] in zj:
            print("您的位置在中间")
        elif seat[1] in gd:
            print("您的位置在过道")
        else:
            print("座位号不存在")
    else:
        print("座位号不存在")
elif len(seat) ==3:
    if 0 < int(seat[0:2]) < 29:
        if seat[2] in ck:
            print("您的位置在窗口")
        elif seat[2] in zj:
            print("您的位置在中间")
        elif seat[2] in gd:
            print("您的位置在过道")
        else:
            print("座位号不存在")
    else:
        print("座位号不存在")
else:
    print("座位号不存在")
