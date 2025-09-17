import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm

# 创建数据集
data = load_digits()
x = data.data
y = data.target

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建模型
clf = GaussianNB()
result = clf.fit(x_train, y_train)
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)
acc_train = clf.score(x_train, y_train)
acc_test = clf.score(x_test, y_test)
pred_prob_train = clf.predict_proba(x_train)
pred_prob_test = clf.predict_proba(x_test)
print(pred_prob_train)
print(pred_prob_test)

# 可视化
cm_train = cm(y_train, y_pred_train)
cm_test = cm(y_test, y_pred_test)
print('Train confusion matrix:\n{}'.format(cm_train))
print('Test confusion matrix:\n{}'.format(cm_test))

plt.plot(y_train, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_pred_train, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Train(Accuracy:{})'.format(acc_train))
plt.show()

plt.plot(y_test, label='y_true', marker='o', markerfacecolor='none', linestyle='')
plt.plot(y_pred_test, label='y_pred', marker='*', linestyle='')
plt.legend()
plt.title('Test(Accuracy:{})'.format(acc_test))
plt.show()
