# 分类
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 加载数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    tf.keras.layers.Dense(units=128, input_dim=784, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型训练
epochs = 20
batch_size = 64
results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# 模型评估
train_loss, train_accuracy = model.evaluate(x_train, y_train)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("train loss:{}, train accuracy:{}".format(train_loss, train_accuracy))
print("test loss:{}, test accuracy:{}".format(test_loss, test_accuracy))

# 可视化
# 损失曲线
plt.plot(results.history["loss"], label="Train Loss")
plt.plot(results.history["val_loss"], label="Test Loss")
plt.legend()
plt.title("Loss curve")
plt.show()

# 准确率
plt.plot(results.history["accuracy"], label="Train Accuracy")
plt.plot(results.history["val_accuracy"], label="Test Accuracy")
plt.legend()
plt.title("Accuracy curve")
plt.show()

# 对比图
y_test_pred = model.predict(x_test)
plt.plot(y_test[:200], label="y_true", marker='o', markerfacecolor='none') # 颜色设置为透明
plt.plot(y_test_pred[:200], label='y_pred', marker='x')
plt.legend()
plt.title("Test")
plt.show()
