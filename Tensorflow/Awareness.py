import tensorflow as tf
import numpy as np

print(tf.__version__)

# 常量、变量
# 创建常量op
op1 = tf.constant([[2, 2]])
op2 = tf.constant([[1], [2]])

# 创建矩阵乘法op
product = tf.matmul(op1, op2)
print(product)

# 直接输出结果
print(product.numpy())  # TensorFlow 2.x 可以直接通过 .numpy() 获取张量的值

# 创建一个变量op
x = tf.Variable([[6, 6]])
c = tf.constant([[1, 2]])

# 创建加法、减法op
add = tf.add(x, c)
sub = tf.subtract(x, c)
print(add.numpy())
print(sub.numpy())

# 变量赋值
state = tf.Variable([0])
for i in range(5):
    operation = tf.add(state, [1])
    state.assign(operation)
    print(state.numpy())
print(state)

# 转换格式
x = tf.constant([[1, 2], [3, 4]])
x.numpy()
print(x.numpy())

y = tf.cast(x, dtype=tf.float32)
print(y)

z = np.array([1, 6])
p = tf.multiply(z, 2)
print(p)
