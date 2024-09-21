#第一题
import csv
data = [
["99100",90,100,91,80],
["99101",89,95,99,80],
["99102",87,90,67,100],
["99103",100,99,95,90],
["99104",78,80,86,88]]
for i in range(5):
    total_grades = sum(data[i][1:])
    data[i].append(total_grades)
head = ["学号","语文","数学","英语","python","总分"]
with open("scores.csv","w",encoding="utf-8",newline="") as file:
    a = csv.writer(file)
    a.writerow(head)
    a.writerows(data)
print("您已将学生的考试成绩保存为csv文件格式！")

#第二题
import jieba
data = "记得那是三年前\n我们来到张贵庄\n拥挤的班车进了民航\n大学里度时光\n忘不了那尘土飞扬\n忘不了那拥挤的二食堂\n忘不了宿舍里蚊子嗡嗡响\n忘不了那多情的姑娘\n哦~\n多少次我失意彷徨\n多少次我走进课堂\n多少次你我并肩在路上\n多少次梦回故乡\n在风中你依偎我身旁\n在风中我为你歌唱\n二公寓的大妈你不要阻挡\n同学情谊长\n明天我就要离开张贵庄\n明天我就要奔向远方\n家乡的姑娘你不要惆怅\n明天我就要回到你身旁\n啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦"
data1 = data.replace("\n","")
dict = {}
words = jieba.cut(data1)
for i in words:
    if i not in dict:
        dict[i] =0
    dict[i] += 1
final = sorted(dict.items(),key=lambda x:x[1],reverse=True)
print(final)
