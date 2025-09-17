from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_breast_cancer
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = load_breast_cancer()

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 创建模型
clf = RandomForestClassifier(n_estimators=34,
                             criterion='entropy',
                             max_depth=12,
                             min_samples_leaf=5
                             )
clf = clf.fit(x_train, y_train)
train_result = clf.score(x_train, y_train)
val_result = cross_val_score(clf, data.data, data.target, cv=10)
val_result_mean = val_result.mean()
test_result = clf.score(x_test, y_test)
print('Train accuracy:{}, Cross validation accuracy:{}, Test accuracy:{}'.format(train_result, val_result_mean, test_result))

feature_importances = [*zip(data.feature_names, clf.feature_importances_)]
print(feature_importances)

# 可视化
# 模型结构
single_tree = clf.estimators_[0]
feature_names = data.feature_names
structure = tree.export_graphviz(single_tree,
                                 feature_names=feature_names,
                                 class_names=['0', '1', '2'],
                                 filled=True,
                                 rounded=True)
graph = graphviz.Source(structure)
graph.view()

# 模型预测
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
plt.plot(y_train, label='y_true')
plt.plot(y_train_pred, label='y_pred')
plt.legend()
plt.title('Train(Accuracy:{})'.format(train_result))
plt.show()

plt.plot(y_test, label='y_true')
plt.plot(y_test_pred, label='y_pred')
plt.legend()
plt.title('Test(Accuracy:{})'.format(test_result))
plt.show()

# # 交叉验证曲线
# cross_val_mean = []
# for i in range(1, 11, 1):
#     val_result = cross_val_score(clf, data.data, data.target, cv=10).mean()
#     cross_val_mean.append(val_result)
#
# plt.plot(range(1, 11, 1), cross_val_mean)
# plt.title('Cross validation curve')
# plt.show()

# # 超参数曲线---调整树的个数
# cross_val_mean = []
# for i in range(1, 101, 1):
#     clf = RandomForestClassifier(n_estimators=i)
#     val_result_mean = cross_val_score(clf, data.data, data.target, cv=10).mean()
#     cross_val_mean.append(val_result_mean)
#
# print('n_estimators:{}, max value:{}'.format(cross_val_mean.index(max(cross_val_mean)), max(cross_val_mean)))
# plt.plot(range(1, 101, 1), cross_val_mean)
# plt.title('Hyperparameter Optimization')
# plt.show()

# # 网格搜索
# clf = RandomForestClassifier(
#                              )
# param_grid = {'n_estimators':np.arange(1, 100, 1)}
# param_grid = {'criterion':['gini', 'entropy']}
# param_grid = {'max_depth':np.arange(1, 20, 1)}
# param_grid = {'min_samples_leaf':np.arange(1, 30, 1)}
# param_grid = {'min_samples_split':np.arange(1, 20, 1)}
# param_grid = {'max_features':np.arange(1, 30, 1)}
# GS = GridSearchCV(clf, param_grid=param_grid, cv=10)
# GS.fit(data.data, data.target)
# print(GS.best_params_)
# print(GS.best_score_)
