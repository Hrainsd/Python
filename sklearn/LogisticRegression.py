import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
data = load_breast_cancer()
x = data.data
y = data.target

# 创建模型
lr1 = LR(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)
lr2 = LR(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)

result1 = lr1.fit(x, y)
result2 = lr2.fit(x, y)
print(result1.coef_) # 参数θ的值
print((result1.coef_ != 0).sum(axis=1))
print(result2.coef_)

# 学习曲线
acc1_train = []
acc1_test = []
acc2_train = []
acc2_test = []
x_label = np.linspace(0.05, 2.0, 30)
for i in x_label:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    lr1 = LR(penalty='l1', C=i, solver='liblinear', max_iter=1000)
    lr2 = LR(penalty='l2', C=i, solver='liblinear', max_iter=1000)
    lr1.fit(x_train, y_train)
    lr2.fit(x_train, y_train)

    acc1_train.append(accuracy_score(y_train, lr1.predict(x_train)))
    acc1_test.append(accuracy_score(y_test, lr1.predict(x_test)))
    acc2_train.append(accuracy_score(y_train, lr2.predict(x_train)))
    acc2_test.append(accuracy_score(y_test, lr2.predict(x_test)))

print(acc1_train)
print(acc1_test)
print(acc2_train)
print(acc2_test)

accs = [acc1_train, acc1_test, acc2_train, acc2_test]
colors = ['r', 'g', 'b', 'c']
label = ['L1 train accuracy', 'L1 test accuracy', 'L2 train accuracy', 'L2 test accuracy']
for i in range(4):
    plt.plot(x_label, accs[i], color=colors[i], label=label[i])
plt.legend(loc='best')
plt.title('C value curves')
plt.show()
