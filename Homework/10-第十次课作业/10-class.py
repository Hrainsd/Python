#文件
file = open("txt1.txt","r",encoding="gb2312") #r表示只读
a = file.read()
a_ = file.read(3)
a__ = file.readline(4)
a___ = file.readlines(2)
print(a)
print(a_)
print(a__)
print(a___)
file.close()

file_ = open("txt2.txt","w",encoding="gb2312") #w表示覆盖写
file_.write("我爱你祖国母亲！")
file_.close()

my_class = ["张三：111111","李四：222222","王五：333333"] #将一个元素全为字符串的列表写入文件
file__ = open("txt3.txt","w",encoding="utf-8")
file__.writelines(my_class)
file__.close()
file__ = open("txt3.txt","r",encoding="utf-8")
b = file__.read()
print(b)
file__.close()

my_class = ["张三：111111","李四：222222","王五：333333"]
file__ = open("txt3.txt","w",encoding="utf-8")
for i in my_class:
    file__.writelines(i+"\n")
file.close()
file__ = open("txt3.txt","r",encoding="utf-8")
c = file__.read()
print(c)
file__.close()

f = open("txt2.txt","a") #a表示追加写，追加到最后
f.write("\n我爱爸爸妈妈!")
f.close()
f_ = open("txt2.txt","r")
d = f_.read()
print(d)
f_.close()

#b表示二进制文件模式，t表示文本文件模式

#文件读取
fn = input("请输入要打开的文件名称：")
fo = open(fn,"r")
txt = fo.read()
fo.close()
fn_ = input("请输入要打开的文件名称：")
fo_ = open(fn_,"r")
txt_ = fo_.read()
while txt_ != "":
    txt_ = fo_.read(2)
print(txt_)
fo_.close()
