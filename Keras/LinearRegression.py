import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# 输入
x = np.random.rand(100)
noise = np.random.normal(0, 0.01, x.shape)

# 输出
y = 0.1 * x + 0.2 + noise

# 可视化：散点图
plt.scatter(x, y)
plt.show()

# 创建模型
model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(optimizer="adam", loss="mse")

# 训练
epochs = 5000
for i in range(epochs):
    cost = model.train_on_batch(x, y)
    if (i + 1) % 100 == 0:
        print("cost:{}".format(cost))

# 打印模型参数
W, b = model.layers[0].get_weights()
print("W:{}, b:{}".format(W, b))

# 模型预测
y_pred = model.predict(x)

# 可视化模型结果
plt.scatter(x, y)
plt.plot(x, y_pred, "r-", lw=3)
plt.show()
