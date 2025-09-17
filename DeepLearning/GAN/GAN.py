import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.family'] = 'SimSun'  # 设置字体为宋体
# 设置字体大小
plt.rcParams['font.size'] = 16

# 加载数据集并进行预处理
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\GAN\1.csv"
df = pd.read_csv(file_path).dropna()
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 定义生成器和判别器
def build_generator(latent_dim, num_features):
    return models.Sequential([
        layers.Dense(256, activation='relu', input_dim=latent_dim),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(2048, activation='relu'),
        layers.Dense(num_features, activation='tanh')
    ])

def build_discriminator(num_features):
    return models.Sequential([
        layers.Dense(1024, activation='relu', input_dim=num_features),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])

# 初始化模型和超参数
latent_dim = 128
batch_size = 128
epochs = 50
clip_value = 0.01

generator = build_generator(latent_dim, df_normalized.shape[1])
discriminator = build_discriminator(df_normalized.shape[1])

# WGAN的损失函数
def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

# 编译判别器和生成器
discriminator.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00005), loss=wasserstein_loss)
generator_input = layers.Input(shape=(latent_dim,))
generated_data = generator(generator_input)
discriminator.trainable = False
gan_output = discriminator(generated_data)
gan = models.Model(generator_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00005), loss=wasserstein_loss)

# 记录损失
d_losses = []
g_losses = []

# 训练WGAN
for epoch in range(epochs):
    for _ in range(32):  # 训练判别器
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
        real_data = df_normalized.sample(batch_size).values
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generator.predict(noise), -np.ones((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Clip判别器权重
        for layer in discriminator.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            layer.set_weights(weights)

    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")

# 生成并保存数据
noise = np.random.normal(0, 1, size=[int(len(df) * 0.2), latent_dim])
generated_data_normalized = generator.predict(noise)
generated_data = scaler.inverse_transform(generated_data_normalized)
output_file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\GAN\2.csv"
pd.DataFrame(generated_data, columns=df.columns).to_csv(output_file_path, index=False)
print(f"Generated data saved to {output_file_path}")

# 保存损失到CSV文件
pd.DataFrame(d_losses, columns=['D Loss']).to_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\GAN\D_loss.csv', index=False)
pd.DataFrame(g_losses, columns=['G Loss']).to_csv(r'C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\GAN\G_loss.csv', index=False)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='判别器损失', color='#A8DADC')
plt.plot(g_losses, label='生成器损失', color='#E4C1F9')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.legend(frameon = False)
# plt.title('损失曲线')
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\GAN\beihe_Loss Curves.svg")
plt.show()
