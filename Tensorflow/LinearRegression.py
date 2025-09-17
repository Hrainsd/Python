import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 创建数据集
x = np.random.rand(100)
noise = np.random.normal(0, 0.02, size=x.shape)
y = 0.5 * x + 0.6 + noise

plt.scatter(x, y)
plt.show()

# 创建模型变量
k = tf.Variable(0.0)
b = tf.Variable(0.0)

# 损失函数
def compute_loss():
    y_hat = k * x + b
    return tf.reduce_mean(tf.square(y_hat - y))

# 优化器
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# 模型训练
epochs = 3000
for i in range(epochs):
    optimizer.minimize(compute_loss, var_list=[k, b])

    # 每100次迭代打印一次结果
    if (i + 1) % 100 == 0:
        loss = compute_loss().numpy()
        print("Iteration{}, loss: {:.4f}, k: {:.4f}, b: {:.4f}".format(i + 1, loss, k.numpy(), b.numpy()))

# 可视化结果
cmap = plt.cm.rainbow
y_pred = k.numpy() * x + b.numpy()
plt.scatter(x, y, cmap=cmap)
plt.plot(x, y_pred, 'c-', label='y_pred', lw=3)
plt.legend()
plt.show()
