#第一题
dic = {"cauc":666666,"kg101":777777,"kg102":888888}
zh = str(input("请输入您的帐号："))
if zh in dic:
    mm = int(input("请输入密码："))
    if mm == dic[zh]:
        print("Success")
    else:
        print("Fail")
else:
    print("Wrong User")

#第二题
dic_ = {"中国民航大学":"天津","中国民航飞行学院":"广汉","民航上海中等专业学校":"上海","中国民航干部管理学院":"北京","广州民航职业技术学院":"广州"}
college_name = input("请输入学校名称：")
if college_name in dic_:
    print(college_name+"所在城市为："+dic_[college_name])
elif college_name == "0":
    print("您已直接退出程序")
else:
    print("输入错误")

#第三题
dic__ = {"袁隆平院士":"16999999999","吴孟超院士":"11999999999","特朗普":"22222222222","普京":"522888666886"}
name = input("请输入添加到通讯录的姓名：")
if name in dic__:
    print("您输入的姓名在通讯录中已存在")
else:
    phone_number = input("请输入该联系人的电话号码：")
    dic__[name] = phone_number
    print(dic__)
