import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC

# 创建数据集
x, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=2)

plt.scatter(x[:, 0], x[:, 1], c=y, marker='o')
plt.show()

colors = ['r', 'g', 'b']
labels = ['Feature1', 'Feature2', 'Feature3']
for i, color, label in zip(range(3), colors, labels):
    plt.scatter(x[y == i, 0], x[y == i, 1], color=color, label=label, alpha=0.2)
plt.legend(loc='best')
plt.show()

clf = SVC(kernel='linear')
result = clf.fit(x, y)
y_pred = clf.predict(x)
acc = clf.score(x, y)
decision_edge_dis = clf.decision_function(x)
print(decision_edge_dis.shape)
print(decision_edge_dis)

plt.plot(y, label='y_true', marker='o', linestyle='')
plt.plot(y_pred, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('SVM(Accuracy:{})'.format(acc))
plt.show()

# 环形数据集
x, y = make_circles(n_samples=500, noise=0.1, factor=0.3)

plt.scatter(x[:, 0], x[:, 1], c=y, marker='o', alpha=0.5)
plt.show()

clf = SVC(kernel='rbf')
result = clf.fit(x, y)
y_pred = clf.predict(x)
acc = clf.score(x, y)
decision_edge_dis = clf.decision_function(x)
print(decision_edge_dis.shape)
print(decision_edge_dis)

plt.plot(y, label='y_true', marker='o', linestyle='')
plt.plot(y_pred, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('SVM(Accuracy:{})'.format(acc))
plt.show()


# 核函数要么选linear，要么选rbf
# C越大，模型训练时间越短；C越小，模型精确率越高
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 加载数据集
data = load_breast_cancer()
x = data.data
y = data.target

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
result = clf.fit(x_train, y_train)
train_acc = clf.score(x_train, y_train)
test_acc = clf.score(x_test, y_test)
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

plt.plot(y_train, label='y_true', marker='o', linestyle='', markerfacecolor='none')
plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Train(Accuracy:{:.4f})'.format(train_acc))
plt.show()

plt.plot(y_test, label='y_true', marker='o', linestyle='', markerfacecolor='none')
plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Test(Accuracy:{:.4f})'.format(test_acc))
plt.show()

# 网格搜索
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

gamma_range = np.logspace(-10, 1, 20) # 产生从10^(-10)到10^(1)之间的20个数
coef0_range = np.linspace(0, 5, 10)
param_grid = {'gamma':gamma_range, 'coef0':coef0_range}

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
clf = SVC(kernel='poly', degree=1)
GS = GridSearchCV(clf, param_grid=param_grid, cv=cv)
GS.fit(x, y)
print('The best parameters are {} with a score of {}'.format(GS.best_params_, GS.best_score_))

# C的学习曲线
acc_C = []
C = np.linspace(0.01, 1, 50)

for i in C:
    clf = SVC(C=i, kernel='linear')
    clf.fit(x_train, y_train)
    acc_C.append(clf.score(x_test, y_test))

print('C = {}, 精确率最大:{:.4f}'.format(acc_C.index(np.max(acc_C)), np.max(acc_C)))
plt.plot(C, acc_C)
plt.title('C learning curve')
plt.show()
