import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# 加载数据
x = np.linspace(-2, 2, num=200)
noise = np.random.normal(0, 0.5, x.shape)

y = np.square(x) + noise

plt.scatter(x, y)
plt.show()

# 优化器
optim = Adam(lr=0.01)

# 创建模型
model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='tanh'))
# model.add(Activation('tanh'))
model.add(Dense(units=1))
# model.add(Activation('tanh'))
model.compile(optimizer=optim, loss='mse')

# 参数设置
epochs = 3000

# 模型训练
for i in range(epochs):
    cost = model.train_on_batch(x, y)
    if (i + 1) % 100 == 0:
        print("cost:", cost)

# 预测
y_pred = model.predict(x)

# 可视化
plt.scatter(x, y)
plt.plot(x, y_pred, 'c-', lw=3)
plt.show()
