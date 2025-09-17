import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 创建数据集
x = np.linspace(-1, 1, 200).reshape(-1, 1)
noise = np.random.normal(0, 0.02, x.shape)
y = np.square(x) + noise

plt.scatter(x, y)
plt.title("Original Data")
plt.show()

# 数据预处理
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='tanh'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# 优化器
learning_rate = 1e-3
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 损失函数
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 训练模型
epochs = 3000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

# 可视化结果
y_pred = model(x).numpy()
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, 'c-', label='Fitted Line', lw=3)
plt.title("Prediction result")
plt.legend()
plt.show()
