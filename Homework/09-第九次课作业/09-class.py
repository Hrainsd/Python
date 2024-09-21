#函数
#def 函数名(参数1, 参数2，*参数，...) :
def sum(a,b):
    result = a+b
    return result
m = eval(input("第一个数："))
n = eval(input("第二个数："))
print(sum(m,n))

n = 5 # 阶乘非函数版
result = 1
for i in range(1,n+1):
    result *= i
print(result)

def jiecheng(n): #阶乘函数版1
    result = 1
    for i in range(1,n+1):
        result *= i
    return result
shu = eval(input("输入一个数："))
result_ = jiecheng(shu)
print(result_)

def jiecheng(n): #阶乘函数版2递归调用
    if n == 1:
        return 1
    else:
        return n*jiecheng(n-1)
shu = eval(input("输入一个数："))
result_ = jiecheng(shu)
print(result_)

def sum_list(numbers): #用列表做参数
    total = 0
    for num in numbers:
        total += num
    return total
list_ = [1,2,3,4,5]
sum_list_ = sum_list(list_)
print(sum_list_)

def count_(string): #字符串做参数
    count_string = set(string)
    return len(count_string)
n = input("输入一个字符串：")
print(count_(n))

def add_(numbers): #字典做参数
    sum = 0
    for num in numbers.values():
        sum += num
    return sum
numbers_ = {"num1":1,"num2":2,"num3":3}
result_ =add_(numbers_)
print(result_)

def sum(*canshu): #元组做参数
    total = 0
    for i in canshu:
        total += i
    return total
result_ = sum(1,2,3,4,5)
print(result_)

def a(n,m=1): #参数的传递
    b = 1
    for i in range(1,n+1):
        b *= i
    return b//m,n,m
print(a(10,5)) #参数的位置传递
print(a(m=5,n=10)) #参数的名称传递
e,f,g = a(10,5)
print(e,f,g)

#lambda函数
square = lambda x:x**2
result = square(6)
print(result)
sum = lambda x,y:x+y
result = sum(6,6)
print(result)

#map函数map(function,object)将函数作用到对象
lb = [1,2,3,4,5,6]
square_lb =list(map(lambda x:x**2,lb))
print(square_lb)

lb_ =[1,2,3,4,5]
lb__ = [6,7,8,9,10]
plus = list(map(lambda x,y:x+y,lb_,lb__))
print(plus)

a,b,c = map(int,input("请输入三个数：").split())
cj = a*b*c
print(cj)
