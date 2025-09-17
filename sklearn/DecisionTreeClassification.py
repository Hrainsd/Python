from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
wine = load_wine()
data = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
print(data)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

# 创建模型
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=3,
                                  min_samples_leaf=5,
                                  )
clf = clf.fit(x_train, y_train)
train_result = clf.score(x_train, y_train)
test_result = clf.score(x_test, y_test)
print(clf.apply(x_train)) # 返回每个样本所在叶子节点的索引
print(clf.apply(x_test))
print('Train accuracy:{}'.format(train_result))
print('Test accuracy:{}'.format(test_result))

# 可视化
# 模型结构
feature_names = wine.feature_names
print(feature_names)
feature_importances = [*zip(wine.feature_names, clf.feature_importances_)]
print(feature_importances)

gra_data = tree.export_graphviz(clf,
                             feature_names=feature_names,
                             class_names=['0', '1', '2'],
                             filled=True,
                             rounded=True)
graph = graphviz.Source(gra_data)
graph.view()

# 模型预测
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
plt.plot(y_train, label='y_true')
plt.plot(y_train_pred, label='y_pred')
plt.legend()
plt.title('Train(Accuracy:{:.4f})'.format(train_result))
plt.show()

plt.plot(y_test, label='y_true')
plt.plot(y_test_pred, label='y_pred')
plt.legend()
plt.title('Test(Accuracy:{:.4f})'.format(test_result))
plt.show()

# 超参数曲线---调参，获取最优超参数
train_para = []
test_para = []
for i in range(5):
    for j in range(20):
        clf = tree.DecisionTreeClassifier(criterion='entropy',
                                           max_depth=i+1,
                                           min_samples_leaf=j+1,
                                           )
        clf = clf.fit(x_train, y_train)
        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        train_para.append(train_acc)
        test_para.append(test_acc)

plt.plot(range(1, 101), train_para, label='Train accuracy')
plt.plot(range(1, 101), test_para, label='Test accuracy')
plt.legend()
plt.title('Hyperparameter Optimization')
plt.show()
