#第一题
file = open("grade.txt","r",encoding="utf-8")
file1 = file.readlines()
list = [int(file1[i])for i in range(len(file1))]
file.close()
file3 = open("result.txt","w",encoding="utf-8")
file3.write("{}：{}\n".format("输出的最高分",max(list)))
file3.write("{}：{}\n".format("输出的最低分",min(list)))
file3.write("{}：{:.1f}".format("输出的平均分",sum(list)/len(list)))
file3.close()
file_ = open("result.txt","r",encoding="utf-8")
file_1 = file_.read()
print(file_1)
file_.close()

#第二题
#1小题
file0 = open("gushi.txt","w")
file0.write("春江花月夜\n张若虚\n春江潮水连海平，海上明月共潮生。\n滟滟随波千万里，何处春江无月明。\n江流宛转绕芳甸，月照花林皆似霰。\n空里流霜不觉飞，汀上白沙看不见。\n江天一色无纤尘，皎皎空中孤月轮。\n江畔何人初见月？江月何年初照人？\n人生代代无穷已，江月年年望相似。\n不知江月待何人，但见长江送流水。\n白云一片去悠悠，青枫浦上不胜愁。\n谁家今夜扁舟子？何处相思明月楼？\n可怜楼上月裴回，应照离人妆镜台。\n玉户帘中卷不去，捣衣砧上拂还来。\n此时相望不相闻，愿逐月华流照君。\n鸿雁长飞光不度，鱼龙潜跃水成文。\n昨夜闲潭梦落花，可怜春半不还家。\n江水流春去欲尽，江潭落月复西斜。\n斜月沉沉藏海雾，碣石潇湘无限路。\n不知乘月几人归，落月摇情满江树。")
file0.close()
#2小题
def readfile():
     file = open(read,"r")
     content = file.read()
     file.close()
     print("读取完毕")
def writefile():
    file_ = open(read, "r")
    content_ = file_.read()
    file_.close()
    file_1 = open("copy.txt","w")
    file_1.write("{}".format(content_))
    file_1.close()
    print("复制完毕")
read = input("请输入您想要读取的指定文件：")
A = readfile()
B = writefile()
