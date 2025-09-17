# 迁移学习---猫狗识别
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
)
val_datagen = ImageDataGenerator(
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
val_data = val_datagen.flow_from_directory(
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
model = tf.keras.applications.VGG16(include_top=False,
                                        weights='imagenet',
                                        input_shape=(img_height, img_width, 3))
# 冻结预训练模型的参数
model.trainable = False
# 自定义全连接层
x = model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=128, activation='relu',, kernel_regularizer='l2')(x)
x = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer='l2')(x)

model = tf.keras.models.Model(model.input, x)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# 模型训练
epochs = 10
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
result = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[callbacks])

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
