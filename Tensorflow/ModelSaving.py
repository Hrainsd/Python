# 模型的保存和读取
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 加载数据
fasion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fasion_mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, 784)/255.0
x_test = x_test.reshape(-1, 784)/255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 优化器
learning_rate = 1e-3
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, input_dim=784, activation='tanh'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 模型训练
batch_size = 64
epochs = 30
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# 模型评估
train_loss, train_accuracy = model.evaluate(x_train, y_train)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"train loss:{train_loss}, train accuracy:{train_accuracy}")
print(f"test loss:{test_loss}, test accuracy:{test_accuracy}")

# 结果可视化
# 对比图
y_test_pred = model.predict(x_test)
plt.plot(y_test[:100], label="y_test", marker='o', markerfacecolor='none')
plt.plot(y_test_pred[:100], label='y_test_pred', marker='x')
plt.legend()
plt.title("Test")
plt.show()

# 模型保存（保存网络结构和参数）
model.save("model.h5")
# 模型读取
model = tf.keras.models.load_model("model.h5")

# 仅保存网络结构
config = model.to_json()
# 模型结构读取
model = tf.keras.models.model_from_json(config)
with open("config.json", 'w') as json:
    json.write(config)
model.summary()

# 仅保存网络参数
weights = model.get_weights()
model.save_weights("model_weights.h5")
# 模型参数读取
model.load_weights("model_weights.h5")
