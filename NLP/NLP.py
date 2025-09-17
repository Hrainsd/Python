# import nltk
# import jieba
# from nltk.corpus import stopwords
#
# # NLP:原始文本——》文本预处理——》文本列表——》特征工程——》vectors——》model——》evaluation
#
# # 英文分词
# sentence = 'life is like a box of chocolate'
# tokens = nltk.word_tokenize(sentence)
# print(tokens)
# # 词性
# pos = nltk.pos_tag(tokens)
# print(pos)
# # 去除停止词
# filterd_word = [word for word in tokens if word not in stopwords.words('english')]
# print(filterd_word)
#
# # 中文分词
# sentence = '今天天气很好，我想去晒太阳'
#
# tokens = jieba.tokenize(sentence)
# tokens_only = [word for word, start, end in tokens]  # 只提取分词的部分
# print('/'.join(tokens_only))
#
# cuts = jieba.cut(sentence, cut_all=False)
# print('/'.join(cuts))
#
# search_cuts = jieba.cut_for_search(sentence)
# print('/'.join(search_cuts))


# # 每日新闻预测股票市场
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import nltk
# from sklearn.svm import SVC
# from sklearn.metrics import roc_auc_score
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from datetime import date
# from nltk.corpus import stopwords
# import re
# from nltk.stem import WordNetLemmatizer
#
# # 加载数据集
# data = pd.read_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\NLP\Combined_News_DJIA.csv')
# print(data.head())
#
# # 数据预处理
# data['combined_news'] = data.filter(regex=('Top.*')).apply(lambda x:''.join(str(x.values)), axis=1)
#
# # 划分数据集
# train_data = data[data['Date'] < '2015-01-01']
# test_data = data[data['Date'] > '2014-12-31']
#
# # 文本预处理
# # tokens
# x_train = train_data['combined_news']
# x_test = test_data['combined_news']
# x_train = x_train.str.lower().str.replace('"', '').str.replace("'", '').str.split()
# x_test = x_test.str.lower().str.replace('"', '').str.replace("'", '').str.split()
# print(x_test[1611])
#
# # 删除停止词
# stop = stopwords.words('english')
#
# # 删除数字
# def hasNumbers(inputString):
#   return bool(re.search(r'\d', inputString))
#
# # 词形归一
# lemmatizer = WordNetLemmatizer()
#
# # 文本预处理函数
# def check(word):
#   # False,去除该单词
#   # True,保留该单词
#   if word in stop:
#     return False
#   elif hasNumbers(word):
#     return False
#   else:
#     return True
#
# x_train = x_train.apply(lambda x: [lemmatizer.lemmatize(item) for item in x if check(item)])
# x_test = x_test.apply(lambda x: [lemmatizer.lemmatize(item) for item in x if check(item)])
# print(x_test[1611])
# x_train = x_train.apply(lambda x:' '.join(x))
# x_test = x_test.apply(lambda x:' '.join(x))
# print(x_test[1611])
#
# # 特征提取
# feature_extraction = TfidfVectorizer()
# x_train = feature_extraction.fit_transform(x_train.values)
# x_test = feature_extraction.transform(x_test.values)
# y_train = train_data['Label'].values
# y_test = test_data['Label'].values
# print(x_test)
#
# # 创建模型
# clf = SVC(probability=True, kernel='rbf')
# clf.fit(x_train, y_train)
#
# # 预测
# y_train_pred = clf.predict(x_train)
# y_test_pred = clf.predict(x_test)
# y_train_prob = clf.predict_proba(x_train)
# y_test_prob = clf.predict_proba(x_test)
#
# # 计算分数
# train_score = clf.score(x_train, y_train)
# test_score = clf.score(x_test, y_test)
# train_ROC_AUC = roc_auc_score(y_train, y_train_prob[:, 1])
# test_ROC_AUC = roc_auc_score(y_test, y_test_prob[:, 1])
# print(f'Train Score: {train_score}, Train ROC AUC: {train_ROC_AUC}')
# print(f'Test Score: {test_score}, Test ROC AUC: {test_ROC_AUC}')



# 家得宝产品搜索相关性
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# 加载数据集
train_data = pd.read_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\NLP\train.csv', encoding='ISO-8859-1')
test_data = pd.read_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\NLP\test.csv', encoding='ISO-8859-1')
product_description = pd.read_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\NLP\product_descriptions.csv', encoding='ISO-8859-1')

data = pd.concat((train_data, test_data), axis=0, ignore_index=True)
print(data.head(), data.shape)

data = pd.merge(data, product_description, how='left', on='product_uid')
print(data.head())

# 词形归一
stemmer = SnowballStemmer('english')
data['product_title'] = [' '.join([stemmer.stem(item) for item in title.lower().split()]) for title in data['product_title']]
data['search_term'] = [' '.join([stemmer.stem(item) for item in term.lower().split()]) for term in data['search_term']]
data['product_description'] = [' '.join([stemmer.stem(item) for item in desc.lower().split()]) for desc in data['product_description']]
print(data['product_title'])

# 特征工程
# query的长度
data['length of query'] = data['search_term'].map(lambda x:len(x.split())).astype(np.int64)
# 重合数(计算 product_title 中出现 search_term 中的单词数量)
data['commons_in_title'] = [
    sum([1 for word in data['search_term'][i].split() if word in data['product_title'][i]])
    for i in range(data.shape[0])
]

data = data.drop(['product_title', 'search_term', 'product_description'], axis=1)
print(data.head())

# 划分训练集和测试集
train_data = data.iloc[:train_data.shape[0], :]
test_data = data.iloc[train_data.shape[0]:, :]

x_train = train_data[['product_uid', 'length of query', 'commons_in_title']].values
x_test = test_data[['product_uid', 'length of query', 'commons_in_title']].values
y_train = train_data['relevance'].values
y_test = test_data['relevance'].values

# 创建模型并训练
reg = RandomForestRegressor()
result = reg.fit(x_train, y_train)
y_train_pred = reg.predict(x_train)
y_test_pred = reg.predict(x_test)

# 模型评估
train_R2 = reg.score(x_train, y_train)
train_MSE = mean_squared_error(y_train, y_train_pred)
train_cro_val_R2 = cross_val_score(reg, x_train, y_train, cv=10).mean()
print('Train: R2:{}, MSE:{}, Cross Validation Score:{}'.format(train_R2, train_MSE, train_cro_val_R2))

# 可视化
plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_train_pred, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Train(R2:{}, MSE:{})'.format(train_R2, train_MSE))
plt.show()

# 提交文件
submission = pd.DataFrame({'id':test_data['id'].values, 'relevance':np.round(y_test_pred, 2)})
submission.to_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\Kaggle\NLP\submission.csv', index=False)
