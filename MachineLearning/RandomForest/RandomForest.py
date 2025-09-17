import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 设置全局字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载CSV数据
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\RandomForest\1.csv"
df = pd.read_csv(file_path, encoding='GBK')

# 准备数据：拆分特征（X）和目标变量（y）
X = df.iloc[:, :-1]  # 所有列除了最后一列是特征
y = df.iloc[:, -1]   # 最后一列是目标变量

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练随机森林分类器模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 打印准确率
print("Training Set Accuracy:", train_accuracy)
print("Testing Set Accuracy:", test_accuracy)

# 打印分类报告
print("\nTraining Set Classification Report:")
print(classification_report(y_train, y_train_pred))

print("\nTesting Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# 绘制特征重要性图
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color="b", align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=0)
plt.xlim([-1, X.shape[1]])
plt.show()

# 绘制训练集混淆矩阵
cm_train = confusion_matrix(y_train, y_train_pred, labels=np.unique(y))
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Training Set Confusion Matrix')
plt.show()

# 绘制测试集混淆矩阵
cm_test = confusion_matrix(y_test, y_test_pred, labels=np.unique(y))
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Testing Set Confusion Matrix')
plt.show()
