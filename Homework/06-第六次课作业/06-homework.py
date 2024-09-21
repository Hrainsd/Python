#第一题
a = input("请输入若干个整数（数与数之间用空格分隔）：")
b = a.split(" ")
c = [int(b[i]) for i in range(len(b))]
max_ = max(c)
min_ = min(c)
sum_ = sum(c)
av_ = (sum_)/len(c)
print("max："+str(max_)+" min："+str(min_)+" av："+str(av_))
#第二题
A = input("请输入若干个整数（数与数之间用空格分隔）：")
B = A.split(" ")
C = [int(B[i]) for i in range(len(B))]
C.sort()
for i in range(len(B)):
    print(C[i],end=" ")
print()
#第三题
for i in range(1,10000):
    q = i//1000
    b = (i//100)%10
    s = (i%100)//10
    g = (i%100)%10
    if q+b+s+g == 9:
        print(i,end=" ")
