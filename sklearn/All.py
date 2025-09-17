# # 决策树
# from sklearn import tree
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# import graphviz
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载数据
# wine = load_wine()
# data = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
# print(data)
#
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)
#
# # 创建模型
# clf = tree.DecisionTreeClassifier(criterion='entropy',
#                                   max_depth=3,
#                                   min_samples_leaf=5,
#                                   )
# clf = clf.fit(x_train, y_train)
# train_result = clf.score(x_train, y_train)
# test_result = clf.score(x_test, y_test)
# print(clf.apply(x_train)) # 返回每个样本所在叶子节点的索引
# print(clf.apply(x_test))
# print('Train accuracy:{}'.format(train_result))
# print('Test accuracy:{}'.format(test_result))
#
# # 可视化
# # 模型结构
# feature_names = wine.feature_names
# print(feature_names)
# feature_importances = [*zip(wine.feature_names, clf.feature_importances_)]
# print(feature_importances)
#
# gra_data = tree.export_graphviz(clf,
#                              feature_names=feature_names,
#                              class_names=['0', '1', '2'],
#                              filled=True,
#                              rounded=True)
# graph = graphviz.Source(gra_data)
# graph.view()
#
# # 模型预测
# y_train_pred = clf.predict(x_train)
# y_test_pred = clf.predict(x_test)
# plt.plot(y_train, label='y_true')
# plt.plot(y_train_pred, label='y_pred')
# plt.legend()
# plt.title('Train(Accuracy:{:.4f})'.format(train_result))
# plt.show()
#
# plt.plot(y_test, label='y_true')
# plt.plot(y_test_pred, label='y_pred')
# plt.legend()
# plt.title('Test(Accuracy:{:.4f})'.format(test_result))
# plt.show()
#
# # 超参数曲线---调参，获取最优超参数
# train_para = []
# test_para = []
# for i in range(5):
#     for j in range(20):
#         clf = tree.DecisionTreeClassifier(criterion='entropy',
#                                            max_depth=i+1,
#                                            min_samples_leaf=j+1,
#                                            )
#         clf = clf.fit(x_train, y_train)
#         train_acc = clf.score(x_train, y_train)
#         test_acc = clf.score(x_test, y_test)
#         train_para.append(train_acc)
#         test_para.append(test_acc)
#
# plt.plot(range(1, 101), train_para, label='Train accuracy')
# plt.plot(range(1, 101), test_para, label='Test accuracy')
# plt.legend()
# plt.title('Hyperparameter Optimization')
# plt.show()



# # 随机森林
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.datasets import load_breast_cancer
# import graphviz
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载数据
# data = load_breast_cancer()
#
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
#
# # 创建模型
# clf = RandomForestClassifier(n_estimators=34,
#                              criterion='entropy',
#                              max_depth=12,
#                              min_samples_leaf=5
#                              )
# clf = clf.fit(x_train, y_train)
# train_result = clf.score(x_train, y_train)
# val_result = cross_val_score(clf, data.data, data.target, cv=10)
# val_result_mean = val_result.mean()
# test_result = clf.score(x_test, y_test)
# print('Train accuracy:{}, Cross validation accuracy:{}, Test accuracy:{}'.format(train_result, val_result_mean, test_result))
#
# feature_importances = [*zip(data.feature_names, clf.feature_importances_)]
# print(feature_importances)
#
# # 可视化
# # 模型结构
# single_tree = clf.estimators_[0]
# feature_names = data.feature_names
# structure = tree.export_graphviz(single_tree,
#                                  feature_names=feature_names,
#                                  class_names=['0', '1', '2'],
#                                  filled=True,
#                                  rounded=True)
# graph = graphviz.Source(structure)
# graph.view()
#
# # 模型预测
# y_train_pred = clf.predict(x_train)
# y_test_pred = clf.predict(x_test)
# plt.plot(y_train, label='y_true')
# plt.plot(y_train_pred, label='y_pred')
# plt.legend()
# plt.title('Train(Accuracy:{})'.format(train_result))
# plt.show()
#
# plt.plot(y_test, label='y_true')
# plt.plot(y_test_pred, label='y_pred')
# plt.legend()
# plt.title('Test(Accuracy:{})'.format(test_result))
# plt.show()
#
# # # 交叉验证曲线
# # cross_val_mean = []
# # for i in range(1, 11, 1):
# #     val_result = cross_val_score(clf, data.data, data.target, cv=10).mean()
# #     cross_val_mean.append(val_result)
# #
# # plt.plot(range(1, 11, 1), cross_val_mean)
# # plt.title('Cross validation curve')
# # plt.show()
#
# # # 超参数曲线---调整树的个数
# # cross_val_mean = []
# # for i in range(1, 101, 1):
# #     clf = RandomForestClassifier(n_estimators=i)
# #     val_result_mean = cross_val_score(clf, data.data, data.target, cv=10).mean()
# #     cross_val_mean.append(val_result_mean)
# #
# # print('n_estimators:{}, max value:{}'.format(cross_val_mean.index(max(cross_val_mean)), max(cross_val_mean)))
# # plt.plot(range(1, 101, 1), cross_val_mean)
# # plt.title('Hyperparameter Optimization')
# # plt.show()
#
# # # 网格搜索
# # clf = RandomForestClassifier(
# #                              )
# # param_grid = {'n_estimators':np.arange(1, 100, 1)}
# # param_grid = {'criterion':['gini', 'entropy']}
# # param_grid = {'max_depth':np.arange(1, 20, 1)}
# # param_grid = {'min_samples_leaf':np.arange(1, 30, 1)}
# # param_grid = {'min_samples_split':np.arange(1, 20, 1)}
# # param_grid = {'max_features':np.arange(1, 30, 1)}
# # GS = GridSearchCV(clf, param_grid=param_grid, cv=10)
# # GS.fit(data.data, data.target)
# # print(GS.best_params_)
# # print(GS.best_score_)



# # 数据预处理和特征工程
# import numpy as np
# import pandas as pd
#
#
# data = np.array([[1 ,5, 7],
#                  [2, 6 ,8],
#                  [5, 2, 1]])
# data = pd.DataFrame(data)
#
# # 计算每列的最大值、最小值、平均值和总和；每行的话，指定axis=1
# max_per_column = data.max()
# min_per_column = data.min()
# mean_per_column = data.mean()
# sum_per_column = data.sum()
#
# # 打印结果
# print("每列的最大值：\n", max_per_column)
# print("每列的最小值：\n", ",".join(map(str, min_per_column)))
# print("每列的平均值：\n", ",".join(map(str, mean_per_column)))
# print("每列的总和：\n", ",".join(map(str, sum_per_column)))
#
# # 归一化
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# fit_data = scaler.fit(data) # 生成min和max
# output1 = scaler.transform(data) # 通过接口导出结果
# print(output1)
#
# # 特征数量巨大，使用partical_fit
# fit_data1 = scaler.partial_fit(data)
# output2 = scaler.transform(data)
# print(output2)
#
# # fit_transform将fit和transform两步化为一步
# output3 = scaler.fit_transform(data)
# conv_to_input = scaler.inverse_transform(output3)
# print(output3)
# print(conv_to_input)
#
# # 标准化
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# fit_data = scaler.fit(data)
# output1 = scaler.transform(data)
# fit_data1 = scaler.partial_fit(data)
# output2= scaler.transform(data)
# output3 = scaler.fit_transform(data)
# conv_to_input = scaler.inverse_transform(output3)
# print(output1)
# print(output2)
# print(output3)
# print(conv_to_input)
# print(scaler.mean_)
# print(scaler.var_)
# print(output3.mean())
# print(output3.var())
#
# # 缺失值处理
# from sklearn.impute import SimpleImputer
# x = [[1, 4, np.nan, 8],
#      [np.nan, 1, 5, 3],
#      [2, np.nan, 8, 8],
#      [6, 2, 1, np.nan]]
#
# # 第0列数据
# x_0 = [row[0] for row in x]
# x_0 = np.array(x_0).reshape(-1, 1)
#
# imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer2 = SimpleImputer(strategy='median')
# imputer3 = SimpleImputer(strategy='constant', fill_value=0)
# imputer4 = SimpleImputer(strategy='most_frequent')
#
# result1 = imputer1.fit_transform(x)
# result2 = imputer2.fit_transform(x)
# result3 = imputer3.fit_transform(x)
# result4 = imputer4.fit_transform(x_0)
# print(result1)
# print(result2)
# print(result3)
# print(result4)
#
# # 编码与哑变量
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import Binarizer
# from sklearn.preprocessing import KBinsDiscretizer
#
#
# # LabelEncoder---用于标签数据
# x = np.array([['a', 'big', np.nan, 'yes'],
#      ['b', 'small', 5, 'no'],
#      ['b', 'small', 8, 'unknow'],
#      ['a', 'small', 1, 'yes']])
# encoder = LabelEncoder()
# result = encoder.fit_transform(x[:, -1])
# conv_to_input = encoder.inverse_transform(result)
# x[:, -1] = result
# print(result)
# print(conv_to_input)
# print(x)
#
# # OrdinalEncoder---用于特征数据
# x = np.array([['a', 'big', np.nan, 'yes'],
#      ['b', 'small', 5, 'no'],
#      ['b', 'small', 8, 'unknow'],
#      ['a', 'small', 1, 'yes']])
# encoder = OrdinalEncoder()
# result = encoder.fit_transform(x[:, [0, 1]])
# conv_to_input = encoder.inverse_transform(result)
# x[:, 0 : 2] = result
# print(result)
# print(conv_to_input)
# print(x)
#
# # OneHotEncoder---one-hot编码，用于特征和标签数据，都可以
# x = np.array([['a', 'big', np.nan, 'yes'],
#      ['b', 'small', 5, 'no'],
#      ['b', 'small', 8, 'unknow'],
#      ['a', 'small', 1, 'yes']])
# encoder = OneHotEncoder(categories='auto')
# result = encoder.fit_transform(x[:, 0 : 2]).toarray()
# conv_to_input = encoder.inverse_transform(result)
# x = np.append(x, x[:, 2 : 4], axis=1)
# x[:, 0 : 4] = result
# print(result)
# print(conv_to_input)
# print(x)
# print(encoder.get_feature_names_out())
#
# x = pd.DataFrame(x, columns=['f_1_a', 'f_1_b', 'f_2_big', 'f_2_small', 'f_3', 'label'])
# x.drop('f_3', axis=1, inplace=True)
# x.columns = ['Feature_a', 'Feature_b', 'Feature_big', 'Feature_small', 'Label']
# print(x)
#
# # 二值化与分段/分箱
# x = np.array([['a', 'big', 4, 'yes'],
#      ['b', 'small', 5, 'no'],
#      ['b', 'small', 8, 'unknow'],
#      ['a', 'small', 1, 'yes']])
# encoder = Binarizer(threshold=5)
# input = x[:, 2].astype(int).reshape(-1, 1)
# result = encoder.fit_transform(input)
# result = result.reshape(-1)
# x[:, 2] = result
# print(result)
# print(x)
#
# x = np.array([['a', 'big', 4, 'yes'],
#      ['b', 'small', 5, 'no'],
#      ['b', 'small', 8, 'unknow'],
#      ['a', 'small', 1, 'yes']])
# encoder = KBinsDiscretizer(n_bins=2, encode='onehot', strategy='quantile')
# input = x[:, 2].astype(int).reshape(-1, 1)
# result = encoder.fit_transform(input).toarray()
# conv_to_input = encoder.inverse_transform(result)
# x = np.insert(x, 4, x[:, 3], axis=1)
# x[:, 2 : 4] = result
# print(result)
# print(conv_to_input)
# print(x)



# # 降维算法
# # PCA、SVD
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
#
# iris = load_iris()
#
# x = iris.data
# y = iris.target
# y_names = iris.target_names
# print(y_names)
#
# pca = PCA(n_components=2)
# result = pca.fit_transform(x)
# print(result)
# print(pca.explained_variance_) # 可解释性方差，越大越重要
# print(pca.explained_variance_ratio_) # 可解释性方差占原始数据总信息量的比值
# print(pca.explained_variance_ratio_.sum()) # 可解释性方差占比之和
#
#
# colors = ['r', 'g', 'b']
# for i, color, y_name in zip([0, 1, 2], colors, y_names):
#     plt.scatter(result[y == i, 0], result[y == i, 1], color=color, label=y_name)
# plt.legend()
# plt.title('PCA of Iris Dataset')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
#
# # 获取最佳维度
# pca1 = PCA() # 不填n_components时，默认为特征数量和样本数量的最小值，一般为特征数量，找到需要降到的维度
# result1 = pca1.fit_transform(x)
# result1_cumsum = np.cumsum(pca1.explained_variance_ratio_)
# print(result1_cumsum)
# plt.plot(range(1, 5), result1_cumsum)
# plt.title('Cumulus Variance Ratio curve')
# plt.show()



# # Logistic Regression逻辑回归---分类算法
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载数据
# data = load_breast_cancer()
# x = data.data
# y = data.target
#
# # 创建模型
# lr1 = LR(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)
# lr2 = LR(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
#
# result1 = lr1.fit(x, y)
# result2 = lr2.fit(x, y)
# print(result1.coef_) # 参数sita的值
# print((result1.coef_ != 0).sum(axis=1))
# print(result2.coef_)
#
# # 学习曲线
# acc1_train = []
# acc1_test = []
# acc2_train = []
# acc2_test = []
# x_label = np.linspace(0.05, 2.0, 30)
# for i in x_label:
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#     lr1 = LR(penalty='l1', C=i, solver='liblinear', max_iter=1000)
#     lr2 = LR(penalty='l2', C=i, solver='liblinear', max_iter=1000)
#     lr1.fit(x_train, y_train)
#     lr2.fit(x_train, y_train)
#
#     acc1_train.append(accuracy_score(y_train, lr1.predict(x_train)))
#     acc1_test.append(accuracy_score(y_test, lr1.predict(x_test)))
#     acc2_train.append(accuracy_score(y_train, lr2.predict(x_train)))
#     acc2_test.append(accuracy_score(y_test, lr2.predict(x_test)))
#
# print(acc1_train)
# print(acc1_test)
# print(acc2_train)
# print(acc2_test)
#
# accs = [acc1_train, acc1_test, acc2_train, acc2_test]
# colors = ['r', 'g', 'b', 'c']
# label = ['L1 train accuracy', 'L1 test accuracy', 'L2 train accuracy', 'L2 test accuracy']
# for i in range(4):
#     plt.plot(x_label, accs[i], color=colors[i], label=label[i])
# plt.legend(loc='best')
# plt.title('C value curves')
# plt.show()



# # 聚类算法
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 创建数据集
# x, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=42)
#
# # facecolor用在plt.scatter(), markerfacecolor用在plt.plot()
# plt.scatter(x[:, 0], x[:, 1], marker='o', color='black', facecolor='none')
# plt.title('Scatter')
# plt.show()
#
# kmeans = KMeans(n_clusters=4, random_state=42)
# y_pred_ = kmeans.fit_predict(x) # fit_predict可以直接得到划分后的聚类簇
# print(y_pred_)
#
# y_pred = kmeans.fit(x)
# print(y_pred.labels_) # 所有样本被划分到的聚类簇
# print(y_pred.cluster_centers_) # 所有聚类簇的质心数据
# print(y_pred.inertia_) # 总距离平方和
#
# y_pred_label = y_pred.labels_
# centroid = y_pred.cluster_centers_
# colors = ['r', 'g', 'b', 'c']
# labels = ['1', '2', '3', '4']
# figure, ax1 = plt.subplots(1) # 画布，子图对象
# for i, color, label in zip(range(4), colors, labels):
#     ax1.scatter(x[y_pred_label == i, 0], x[y_pred_label == i, 1], marker='o', color=color, s=30, label=label, alpha=0.2)
#     ax1.scatter(centroid[i, 0], centroid[i, 1], marker='*', color=color, s=60)
# plt.legend(loc='best')
# plt.title('Kmeans Scatter')
# plt.show()
#
# # 轮廓系数（越接近1，聚类越合适，也要根据特定的任务去看聚类是否合适）
# from sklearn.metrics import silhouette_score, silhouette_samples
#
# result1 = silhouette_score(x, y_pred_label) # 平均轮廓系数
# result2 = silhouette_samples(x, y_pred_label) # 每个样本的轮廓系数
# print(result1)
# print(result2)



# # SVM支持向量机
# from sklearn.datasets import make_blobs, make_circles
# from sklearn.svm import SVC
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 创建数据集
# x, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=2)
#
# plt.scatter(x[:, 0], x[:, 1], c=y, marker='o')
# plt.show()
#
# colors = ['r', 'g', 'b']
# labels = ['Feature1', 'Feature2', 'Feature3']
# for i, color, label in zip(range(3), colors, labels):
#     plt.scatter(x[y == i, 0], x[y == i, 1], color=color, label=label, alpha=0.2)
# plt.legend(loc='best')
# plt.show()
#
# clf = SVC(kernel='linear')
# result = clf.fit(x, y)
# y_pred = clf.predict(x)
# acc = clf.score(x, y)
# decision_edge_dis = clf.decision_function(x)
# print(decision_edge_dis.shape)
# print(decision_edge_dis)
#
# plt.plot(y, label='y_true', marker='o', linestyle='')
# plt.plot(y_pred, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('SVM(Accuracy:{})'.format(acc))
# plt.show()
#
# # 环形数据
# x, y = make_circles(n_samples=500, noise=0.1, factor=0.3)
#
# plt.scatter(x[:, 0], x[:, 1], c=y, marker='o', alpha=0.5)
# plt.show()
#
# clf = SVC(kernel='rbf')
# result = clf.fit(x, y)
# y_pred = clf.predict(x)
# acc = clf.score(x, y)
# decision_edge_dis = clf.decision_function(x)
# print(decision_edge_dis.shape)
# print(decision_edge_dis)
#
# plt.plot(y, label='y_true', marker='o', linestyle='')
# plt.plot(y_pred, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('SVM(Accuracy:{})'.format(acc))
# plt.show()



# # SVM支持向量机
# # 核函数要么选linear，要么选rbf
# # C越大，模型训练时间越小；C越小，模型精确率越高
# from sklearn.datasets import load_breast_cancer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载数据集
# data = load_breast_cancer()
# x = data.data
# y = data.target
#
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# clf = SVC(kernel='linear')
# result = clf.fit(x_train, y_train)
# train_acc = clf.score(x_train, y_train)
# test_acc = clf.score(x_test, y_test)
# y_pred_train = clf.predict(x_train)
# y_pred_test = clf.predict(x_test)
#
# plt.plot(y_train, label='y_true', marker='o', linestyle='', markerfacecolor='none')
# plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(Accuracy:{:.4f})'.format(train_acc))
# plt.show()
#
# plt.plot(y_test, label='y_true', marker='o', linestyle='', markerfacecolor='none')
# plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(Accuracy:{:.4f})'.format(test_acc))
# plt.show()
#
# # 网格搜索
# from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
#
# gamma_range = np.logspace(-10, 1, 20) # 产生从10^(-10)到10^(1)之间的20个数
# coef0_range = np.linspace(0, 5, 10)
# param_grid = {'gamma':gamma_range, 'coef0':coef0_range}
#
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# clf = SVC(kernel='poly', degree=1)
# GS = GridSearchCV(clf, param_grid=param_grid, cv=cv)
# GS.fit(x, y)
# print('The best parameters are {} with a score of {}'.format(GS.best_params_, GS.best_score_))
#
# # 对C的学习曲线
# acc_C = []
# C = np.linspace(0.01, 1, 50)
#
# for i in C:
#     clf = SVC(C=i, kernel='linear')
#     clf.fit(x_train, y_train)
#     acc_C.append(clf.score(x_test, y_test))
#
# print('C = {}, 精确率最大:{:.4f}'.format(acc_C.index(np.max(acc_C)), np.max(acc_C)))
# plt.plot(C, acc_C)
# plt.title('C learning curve')
# plt.show()



# # 多元线性回归
# # loss为负值，是因为它是一种损失，sklearn给它加了负号，真正的loss是去掉负号后的值
# # R方越接近1越好
# from sklearn.datasets import fetch_california_housing as calif
# from sklearn.linear_model import LinearRegression as LR
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.metrics import r2_score as r2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载数据集
# data = calif()
# x = data.data
# y = data.target
#
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # 创建模型
# lr = LR()
# result = lr.fit(x_train, y_train)
# y_pred_train = lr.predict(x_train)
# y_pred_test = lr.predict(x_test)
# train_R2 = lr.score(x_train, y_train) # 计算结果是R方
# test_R2 = lr.score(x_test, y_test)
# train_MSE = MSE(y_train, y_pred_train)
# test_MSE = MSE(y_test, y_pred_test)
# cross_val = cross_val_score(lr, x, y, cv=10, scoring='neg_mean_squared_error')
# R2_train = r2(y_train, y_pred_train)
# R2_test = r2(y_test, y_pred_test)
# print('Train MSE:{}, Test MSE:{}'.format(train_MSE, test_MSE))
# print('Cross valldation score:{}'.format(cross_val))
# print('Train R2:{}, Test R2:{}'.format(R2_train, R2_test))
# print('截距项：{}'.format(lr.intercept_))
# print([*zip(data.feature_names, lr.coef_)])
# print('\n'.join('Feature name:{}, w:{}'.format(name, weight) for name, weight in zip(data.feature_names, lr.coef_)))
#
# # 可视化
# plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
# plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='', alpha=0.4)
# plt.legend()
# plt.title('Train(MSE:{:.4f}, R2:{:.4f})'.format(train_MSE, train_R2))
# plt.show()
#
# plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
# plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='', alpha=0.4)
# plt.legend()
# plt.title('Test(MSE:{:.4f}, R2:{:.4f})'.format(test_MSE, test_R2))
# plt.show()
#
# plt.plot(sorted(y_train), label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
# plt.plot(sorted(y_pred_train), label='y_pred', marker='*', linestyle='', alpha=0.4)
# plt.legend()
# plt.title('Test(MSE:{:.4f}, R2:{:.4f})'.format(test_MSE, test_R2))
# plt.show()
#
# plt.plot(sorted(y_test), label='y_true', marker='o', markerfacecolor='none', linestyle='', alpha=0.2)
# plt.plot(sorted(y_pred_test), label='y_pred', marker='*', linestyle='', alpha=0.4)
# plt.legend()
# plt.title('Test(MSE:{:.4f}, R2:{:.4f})'.format(test_MSE, test_R2))
# plt.show()


# # KNN---K近邻算法
# # K一般取单数
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix as cm
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载数据集
# data = load_breast_cancer()
# x = data.data
# y = data.target
#
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# clf = KNeighborsClassifier(n_neighbors=5)
# result = clf.fit(x_train, y_train)
# y_pred_train = clf.predict(x_train)
# y_pred_test = clf.predict(x_test)
# acc_train = clf.score(x_train, y_train)
# acc_test = clf.score(x_test, y_test)
# pred_prob_train = clf.predict_proba(x_train)
# pred_prob_test = clf.predict_proba(x_test)
# print(pred_prob_train)
# print(pred_prob_test)
#
# # 可视化
# cm_train = cm(y_train, y_pred_train)
# cm_test = cm(y_test, y_pred_test)
# print('Train confusion matrix:\n{}'.format(cm_train))
# print('Test confusion matrix:\n{}'.format(cm_test))
#
# plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(Accuracy:{})'.format(acc_train))
# plt.show()
#
# plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(Accuracy:{})'.format(acc_test))
# plt.show()



# # NB朴素贝叶斯---分类算法
# from sklearn.naive_bayes import GaussianNB
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix as cm
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 创建数据集
# data = load_digits()
# x = data.data
# y = data.target
#
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # 创建模型
# clf = GaussianNB()
# result = clf.fit(x_train, y_train)
# y_pred_train = clf.predict(x_train)
# y_pred_test = clf.predict(x_test)
# acc_train = clf.score(x_train, y_train)
# acc_test = clf.score(x_test, y_test)
# pred_prob_train = clf.predict_proba(x_train)
# pred_prob_test = clf.predict_proba(x_test)
# print(pred_prob_train)
# print(pred_prob_test)
#
# # 可视化
# cm_train = cm(y_train, y_pred_train)
# cm_test = cm(y_test, y_pred_test)
# print('Train confusion matrix:\n{}'.format(cm_train))
# print('Test confusion matrix:\n{}'.format(cm_test))
#
# plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(Accuracy:{})'.format(acc_train))
# plt.show()
#
# plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(Accuracy:{})'.format(acc_test))
# plt.show()



# # XGBoost
# from xgboost import XGBRegressor as xgb
# from sklearn.datasets import fetch_california_housing as calif
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.metrics import mean_squared_error as MSE
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 加载数据集
# data = calif()
# x = data.data
# y = data.target
# print('Data shape:{}'.format(x.shape))
# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # 创建模型
# reg = xgb(n_estimators=162, learning_rate=0.1)
# result = reg.fit(x_train, y_train)
# y_pred_train = reg.predict(x_train)
# y_pred_test = reg.predict(x_test)
# score_train = reg.score(x_train, y_train) # R方
# score_test = reg.score(x_test, y_test)
# mse_train = MSE(y_train, y_pred_train)
# mse_test = MSE(y_test, y_pred_test)
# feature_importances = reg.feature_importances_
#
# # 交叉验证的结果和.score的结果是相同类型的，比如都是R2，都是准确率等等
# cross_val_R2 = cross_val_score(reg, x_train, y_train, cv=10).mean() # R2
# cross_val_MSE = cross_val_score(reg, x_train, y_train, scoring='neg_mean_squared_error', cv=10).mean() # 使用均方误差作为评估指标
# print(*zip(data.feature_names, feature_importances))
# print('Cross validation score:R2:{:.4f}, MSE:{:.4f}'.format(cross_val_R2, cross_val_MSE))
# kfold = KFold(n_splits=10, shuffle=False)
# cross_val_R2_kfold = cross_val_score(reg, x_train, y_train, cv=kfold).mean() # R2
# print('Cross validation score:KFold:{:.4f}'.format(cross_val_R2_kfold))
#
# # 可视化
# # 学习曲线
# R2_values = []
# MSE_values = []
# for i in range(1, 501, 1):
#     reg = xgb(n_estimators=i)
#     result = reg.fit(x_train, y_train)
#     y_pred_test = reg.predict(x_test)
#     R2_values.append(reg.score(x_test, y_test))
#     MSE_values.append(MSE(y_test, y_pred_test))
# print('n_estimator:{}, Max R2:{}'.format(R2_values.index(max(R2_values)), max(R2_values)))
# print('n_estimator:{}, Min MSE:{}'.format(MSE_values.index(min(MSE_values)), min(MSE_values)))
# plt.plot(R2_values, label='R2', marker='o')
# plt.plot(MSE_values, label='MSE', marker='o')
# plt.title('Learning curve')
# plt.show()
#
# plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(R2:{:.4f}, MSE:{:.4f})'.format(score_train, mse_train))
# plt.show()
#
# plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(R2:{:.4f}, MSE:{:.4f})'.format(score_test, mse_test))
# plt.show()



# # NN神经网络
# # MLP多层感知机
# from sklearn.neural_network import MLPRegressor as MLPreg
# from sklearn.datasets import fetch_california_housing as calif
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # 加载数据集
# data = calif()
# x = data.data
# y = data.target

# # 数据预处理---标准化
# stdscaler = StandardScaler()
# x = stdscaler.fit_transform(x)

# # 划分数据集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # 创建模型
# # activation must be a str among {'relu', 'tanh', 'identity', 'logistic'}
# reg = MLPreg(hidden_layer_sizes=(32, 64, 128), activation='relu', solver='adam')
# result = reg.fit(x_train, y_train)
# y_pred_train = reg.predict(x_train)
# y_pred_test = reg.predict(x_test)
# score_train = reg.score(x_train, y_train)
# score_test = reg.score(x_test, y_test)
# MSE_train = MSE(y_train, y_pred_train)
# MSE_test = MSE(y_test, y_pred_test)
# print('Train:R2:{}, MSE:{}'.format(score_train, MSE_train))
# print('Test:R2:{}, MSE:{}'.format(score_test, MSE_test))

# # 可视化
# plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Train(R2:{:.4f}, MSE:{:.4f})'.format(score_train, MSE_train))
# plt.show()

# plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
# plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
# plt.legend()
# plt.title('Test(R2:{:.4f}, MSE:{:.4f})'.format(score_test, MSE_test))
# plt.show()

# # tanh
# # Train:R2:0.8439893245238443, MSE:0.20855167178307588
# # Test:R2:0.8083279458455801, MSE:0.25116878064182785

# # relu
# # Train:R2:0.8732994047082115, MSE:0.16937059520681286
# # Test:R2:0.7949137321204569, MSE:0.2687468867432714

# # logistic
# # Train:R2:0.7550250595893062, MSE:0.3274771627754287
# # Test:R2:0.7431148834842862, MSE:0.33662456305865096
