# 卷积神经网络---猫狗识别
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 定义图像大小和批次大小
img_height, img_width = 64, 64
batch_size = 64

# 数据集目录
data_dir = r"D:\大学\本科\计算机\人工智能\机器学习\深度学习框架--Tensorflow\dogs-vs-cats\dataset"

# 数据预处理
# 使用 ImageDataGenerator 对图像进行实时数据增强
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# 生成训练数据
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
)
# 生成验证数据
val_data = test_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# 优化器
learning_rate = 1e-3
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2')
])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# 模型训练
epochs = 20
result = model.fit(train_data, epochs=epochs, validation_data=val_data)

# 结果可视化
# 损失曲线
plt.plot(result.history['loss'], label='Train loss')
plt.plot(result.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Loss curves')
plt.show()

# 准确率曲线
plt.plot(result.history['acc'], label='Train accuracy')
plt.plot(result.history['val_acc'], label='Validation accuracy')
plt.legend()
plt.title('Accuracy curves')
plt.show()
