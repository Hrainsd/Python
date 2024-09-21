#第三方库
import csv
with open("plane.csv","w",newline="") as file:
    a = csv.writer(file)
    a.writerow(["飞机","国家"])
    a.writerow(["comac919","中国"])
    a.writerow(["airbus380","法国"])
    a.writerow(["boeing747","美国"])

data_head = ["飞机","国家"]
data = [
["comac919","中国"],
["airbus380","法国"],
["boeing747","美国"]]
with open("plane2.csv","w",newline="") as file:
    b = csv.writer(file)
    b.writerow(data_head)
    b.writerows(data)

#使用两层for循环遍历元素
data = [
["一",90,91,92,93,94,95],
["二",80,81,82,83,84,85],
["三",60,61,62,63,64,65]]
for i in range(3):
    total = sum(data[i][1:8])
    data[i].append(total)
print(data)

#jieba库
import jieba
text = "我是一只猫，快乐的星猫。大角牛，勇敢向前。"
words = jieba.cut(text)
word_dict = {}
for word in words:
    if word not in word_dict:
        word_dict[word] = 0
    word_dict[word] += 1
print(word_dict)

sentence = "我爱你有种左灯右行的冲动"
a = jieba.cut(sentence)
print("/".join(a))

# jieba.lcut(s)是精确模式 jieba.lcut(s,cut_all=True)是全模式
