import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from datetime import datetime

# 加载数据集
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
users = pd.read_csv('users.csv')
events = pd.read_csv('events.csv')
user_friends = pd.read_csv('user_friends.csv')

# 添加一个标志列以区分训练集和测试集
train['is_train'] = 1
test['is_train'] = 0

# 合并训练集和测试集
data = pd.concat([train, test], ignore_index=True)

# 合并用户和事件信息到数据集中
data = data.merge(users, left_on='user', right_on='user_id', how='left')
data = data.merge(events, left_on='event', right_on='event_id', how='left')

# 处理时间戳
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
data['joinedAt'] = pd.to_datetime(data['joinedAt'], errors='coerce')
data['start_time'] = pd.to_datetime(data['start_time'], errors='coerce')

# 提取时间特征
data['event_month'] = data['start_time'].dt.month
data['event_day'] = data['start_time'].dt.day
data['event_hour'] = data['start_time'].dt.hour
data['day_of_week'] = data['start_time'].dt.dayofweek
data['day_of_year'] = data['start_time'].dt.dayofyear

# 判断是否为周末
data['is_weekend'] = data['start_time'].dt.dayofweek >= 5

# 将性别和其他类别特征进行编码
data['gender'] = data['gender'].map({'male': 1, 'female': 0})

# 处理用户好友数据
user_friends['friends'] = user_friends['friends'].apply(lambda x: x.split() if pd.notna(x) else [])
data = data.merge(user_friends, left_on='user', right_on='user', how='left')

# 填补缺失值
data = data.fillna(0)

# 用户历史活动特征
user_activity = train.groupby('user')['event'].count().reset_index(name='user_event_count')
data = data.merge(user_activity, left_on='user', right_on='user', how='left')

# 事件受欢迎程度特征
event_popularity = train.groupby('event')['interested'].mean().reset_index(name='event_popularity')
data = data.merge(event_popularity, left_on='event', right_on='event', how='left')

# 提取目标变量和特征
y_train = data.loc[data['is_train'] == 1, 'interested']
features = ['invited', 'event_month', 'event_day', 'event_hour', 'day_of_year', 'day_of_week', 'gender', 'birthyear', 'timezone', 'lat', 'lng', 'is_weekend', 'user_event_count', 'event_popularity']
X = data[features]
X_train = X.loc[data['is_train'] == 1]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 建立模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 输出特征重要性（特征选择前）
feature_importances_before = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
print("Feature importances before feature selection:")
print(feature_importances_before)

# 验证集预测
y_val_pred = model.predict_proba(X_val)[:, 1]

# 计算MAP@200
def mapk(y_true, y_pred, k=200):
    return average_precision_score(y_true, y_pred)

print(f'MAP@200: {mapk(y_val, y_val_pred)}')

# 特征选择
selector = SelectFromModel(model, threshold='mean', prefit=True)
X_train_selected = selector.transform(X_train)

# 确保测试集数据没有缺失值
X_test = X.loc[data['is_train'] == 0]
X_test = X_test.fillna(0)  # 填补缺失值
X_test_selected = selector.transform(X_test)

# 使用新的特征训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# 输出特征重要性（特征选择后）
feature_importances_after = pd.DataFrame({
    'feature': [features[i] for i in selector.get_support(indices=True)],
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
print("Feature importances after feature selection:")
print(feature_importances_after)

# 验证集预测
X_val_selected = selector.transform(X_val)
y_val_pred = model.predict_proba(X_val_selected)[:, 1]

# 计算MAP@200
print(f'MAP@200: {mapk(y_val, y_val_pred)}')

# 处理测试集
test['interested_pred'] = model.predict_proba(X_test_selected)[:, 1]

# 生成提交文件
submission = test[['user', 'event', 'interested_pred']].copy()
submission['rank'] = submission.groupby('user')['interested_pred'].rank(ascending=False)
submission = submission[submission['rank'] <= 200]
submission = submission.groupby('user')['event'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

# 保存提交文件
submission.to_csv('submission.csv', index=False)
