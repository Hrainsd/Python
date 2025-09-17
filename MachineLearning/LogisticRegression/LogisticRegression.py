# 分类模型---逻辑回归
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# 加载具有特定编码的 CSV 数据（例如，'latin1'）
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\LogisticRegression\1.1.1.csv"
df = pd.read_csv(file_path, encoding='latin1')

# 分离特征（自变量）和目标变量
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练 logistic 回归模型
model = LogisticRegression(multi_class='auto', max_iter=100)
model.fit(X_train_scaled, y_train)

# 预测测试集
y_pred = model.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 计算查准率
precision = precision_score(y_test, y_pred, average='weighted')
print("查准率:", precision)

# 计算召回率
recall = recall_score(y_test, y_pred, average='weighted')
print("召回率:", recall)

# 计算 F1 值
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 值:", f1)

# 绘制混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
plt.xlabel('预测值')
plt.ylabel('真实值')
for i in range(len(np.unique(y))):
    for j in range(len(np.unique(y))):
        plt.text(j, i, conf_mat[i, j], ha='center', va='center', color='white')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体
plt.savefig('Confusion matrix.svg', format='svg', bbox_inches='tight')
plt.show()

# 计算训练集预测准确率
train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))

# 训练集预测结果对比
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_train)), y_train, marker='', linestyle='-', color='#F7C59F', label='真实值')
plt.plot(range(len(y_train)), model.predict(X_train_scaled), marker='o', linestyle='-', color='#A4D4AE', label='预测值')
plt.title(f'训练集预测结果对比 \n准确率: {train_accuracy*100:.2f}%')
plt.xlabel('样本序号')
plt.ylabel('标签值')
plt.legend(frameon=False, bbox_to_anchor=(1.15, 1))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体
plt.savefig('Train.svg', format='svg', bbox_inches='tight')
plt.show()

# 计算测试集预测准确率
test_accuracy = accuracy_score(y_test, y_pred)

# 测试集预测结果对比
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, marker='', linestyle='-', color='#F7C59F', label='真实值')
plt.plot(range(len(y_test)), y_pred, marker='o', linestyle='-', color='#A4D4AE', label='预测值')
plt.title(f'测试集预测结果对比 \n准确率: {test_accuracy*100:.2f}%')
plt.xlabel('样本序号')
plt.ylabel('标签值')
plt.legend(frameon=False, bbox_to_anchor=(1.15, 1))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用宋体
plt.savefig('Test.svg', format='svg', bbox_inches='tight')
plt.show()
