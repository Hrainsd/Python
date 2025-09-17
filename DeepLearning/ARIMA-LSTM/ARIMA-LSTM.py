# ARIMA 用SPSS做即可得到最优模型
# ARIMA-LSTM 利用ARIMA的预测得到ARIMA残差（线性部分） + LSTM输入ARIMA残差预测得到LSTM残差（非线性部分） => 改进的预测值 = 真实值 + LSTM残差
# 使用模型时需要修改的地方：
# 1.文件名 data = pd.read_csv('2016年至今的上证指数.csv', skiprows=1, names=['日期', '指数'])等
# 2.训练数据和测试数据 train_data = data.iloc[:857] test_data = data.iloc[857:]等
# 3.ARIMA参数(p,d,q) model = ARIMA(train_data['data_value'], order=(0,1,14))
# 4.预测多少步 forecast = fitted_model.forecast(steps=20)、forecast_steps = 20
# 5.模型参数 time_steps = 1, model = Sequential()、model.add()等

# 第一步，时间序列图、一阶差分图、二阶差分图
import pandas as pd
import matplotlib.pyplot as plt

# 添加中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取数据，跳过第一行，假设日期列为'日期'，指数列为'指数'
data = pd.read_csv('2016年至今的上证指数.csv', skiprows=1, names=['日期', '指数'])

# 将日期列转换为日期时间格式，假设日期格式为年-月-日
data['日期'] = pd.to_datetime(data['日期'], format='%Y-%m-%d')

# 设置日期列为索引
data.set_index('日期', inplace=True)

# 绘制时序图
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['指数'], label='上证指数', color='#7FFFD4')
plt.title('2016年至今上证指数时序图')
plt.xlabel('日期')
plt.ylabel('指数')
plt.legend(frameon=False)
plt.grid(False)
plt.tight_layout()
plt.savefig('时序图.svg', format='svg', bbox_inches='tight')
plt.show()

# 计算一阶差分
data['一阶差分'] = data['指数'].diff()

# 绘制一阶差分图
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['一阶差分'], label='一阶差分', color='#7FFFD4')
plt.title('2016年至今上证指数一阶差分图')
plt.xlabel('日期')
plt.ylabel('一阶差分')
plt.legend(frameon=False)
plt.grid(False)
plt.tight_layout()
plt.savefig('一阶差分图.svg', format='svg', bbox_inches='tight')
plt.show()

# 计算二阶差分
data['二阶差分'] = data['一阶差分'].diff()

# 绘制二阶差分图
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['二阶差分'], label='二阶差分', color='#7FFFD4')
plt.title('2016年至今上证指数二阶差分图')
plt.xlabel('日期')
plt.ylabel('二阶差分')
plt.legend(frameon=False)
plt.grid(False)
plt.tight_layout()
plt.savefig('二阶差分图.svg', format='svg', bbox_inches='tight')
plt.show()

# 第二步，利用SPSS专家建模器得到最优模型，将ARIMA模型的(p,d,q)输入，使用ARIMA模型拟合并预测
import numpy  as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取CSV文件并存储到DataFrame中
data = pd.read_csv('2016年至今的上证指数.csv')

# 将时间_index列转换为日期时间类型
data['time_index'] = pd.to_datetime(data['time_index'])

# 设置时间_index列为DataFrame的索引
data.set_index('time_index', inplace=True)

# 提取前857个数据作为训练数据，后20个数据作为预测数据
train_data = data.iloc[:857]
test_data = data.iloc[857:]

# 拟合ARIMA模型
model = ARIMA(train_data['data_value'], order=(0,1,14))
fitted_model = model.fit()

# 获取第一个原始数据值
first_original_value = data['data_value'].iloc[0]

# 将拟合数据的第一个值设置为原始数据的第一个值
fitted_values = fitted_model.fittedvalues.copy()
fitted_values.iloc[0] = first_original_value

# 使用拟合的模型进行未来20个数据点的预测
forecast = fitted_model.forecast(steps=20)

# 将拟合数据和原始数据（训练）拼接，计算差值并保存为1.csv
combined_data_1 = pd.concat([data.iloc[:857], pd.DataFrame(fitted_values, columns=['Fitted Data'])], axis=1)
combined_data_1['Diff'] = combined_data_1['data_value'] - combined_data_1['Fitted Data']
combined_data_1.to_csv('1.csv')

# 将预测数据和原始数据（测试）拼接，计算差值并保存为1.csv
forecast_df = pd.DataFrame({'Forecast Data': forecast.values}, index=test_data.index)
combined_data_2 = pd.concat([test_data, forecast_df], axis=1)
combined_data_2['Diff'] = combined_data_2['data_value'] - combined_data_2['Forecast Data']
combined_data_2.to_csv('2.csv')

# Evaluation metrics for training set
train_predictions = fitted_values
train_actual = train_data['data_value']
train_residuals = train_actual - train_predictions

train_mae = mean_absolute_error(train_actual, train_predictions)
train_mse = mean_squared_error(train_actual, train_predictions)
train_mape = np.mean(np.abs(train_residuals / train_actual)) * 100
train_mbe = np.mean(train_residuals)
train_r2 = r2_score(train_actual, train_predictions)

print("Training Set Metrics:")
print(f"MAE: {train_mae}")
print(f"MSE: {train_mse}")
print(f"MAPE: {train_mape}")
print(f"MBE: {train_mbe}")
print(f"R2: {train_r2}")

# Evaluation metrics for forecasted values
forecast_predictions = forecast.values
forecast_actual = test_data['data_value'].values
forecast_residuals = forecast_actual - forecast_predictions

forecast_mae = mean_absolute_error(forecast_actual, forecast_predictions)
forecast_mse = mean_squared_error(forecast_actual, forecast_predictions)
forecast_mape = np.mean(np.abs(forecast_residuals / forecast_actual)) * 100
forecast_mbe = np.mean(forecast_residuals)
forecast_r2 = r2_score(forecast_actual, forecast_predictions)

print("\nForecast Metrics:")
print(f"MAE: {forecast_mae}")
print(f"MSE: {forecast_mse}")
print(f"MAPE: {forecast_mape}")
print(f"MBE: {forecast_mbe}")
print(f"R2: {forecast_r2}")

# 可视化原始数据、拟合数据和预测数据
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['data_value'], label='Original Data', color='#7FFFD4')
plt.plot(train_data.index, fitted_values, label='Fitted Data', linestyle='--', color='#FFCAD4')
plt.plot(test_data.index, forecast, label='Forecast Data', linestyle='--', color='#F64E60')
plt.title('ARIMA(0,1,14) Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('ARIMA结果.svg', format='svg', bbox_inches='tight')
plt.show()

# 第三步，将ARIMA模型训练集的残差作为输入，经过LSTM模型训练后得到拟合的残差以及预测的残差，LSTM拟合残差和预测残差与真实值相加便得到拟合值和预测值
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate ARIMA model's residuals
residuals_train = combined_data_1['Diff'].values

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
residuals_train = scaler.fit_transform(residuals_train.reshape(-1, 1))

# Prepare training data
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

# Set time steps (window size)
time_steps = 1

# Build input-output data
X_train, y_train = create_dataset(residuals_train, time_steps)

# Reshape the input data to [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, 1)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))

optimizer = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
# verbose=0：静默模式，不输出训练过程信息
# verbose=1：默认模式，输出进度条，显示训练过程中的进度和指标
# verbose=2：详细模式，不仅输出进度条，还会显示每个epoch的详细信息，包括损失和指标等
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)

# 使用训练好的模型进行预测
predicted_residuals = model.predict(X_train)

# 逆转换预测的残差数据
predicted_residuals = scaler.inverse_transform(predicted_residuals)
# 将predicted_residuals在第一个位置添加一个值
first_residual_value = combined_data_1['Diff'].values[0]
predicted_residuals = np.insert(predicted_residuals, 0, first_residual_value)

# 得到最终拟合值
predicted_residuals = [[residual] for residual in predicted_residuals]
fitted_values_lstm = train_data.values + predicted_residuals
fitted_values_lstm = np.array(fitted_values_lstm)

# Define the number of steps to forecast
forecast_steps = 20

# Create a copy of the last known residual values to use for forecasting
forecast_residuals = residuals_train[-time_steps:].copy()

# Store forecasted residuals
forecasted_residuals = []

# Generate forecasts
for _ in range(forecast_steps):
    # Reshape the forecast_residuals to match the input shape
    forecast_input = np.reshape(forecast_residuals, (1, time_steps, 1))

    # Predict the next residual value
    next_residual = model.predict(forecast_input)

    # Inverse transform the predicted residual
    next_residual = scaler.inverse_transform(next_residual)

    # Store the predicted residual
    forecasted_residuals.append(next_residual[0, 0])

    # Update the historical residual data by appending the predicted residual
    forecast_residuals = np.append(forecast_residuals[1:], next_residual)

# 得到最终拟合值
forecasted_residuals = [[residual] for residual in forecasted_residuals]
forecasted_values_lstm = test_data.values + forecasted_residuals
forecasted_values_lstm = np.array(forecasted_values_lstm)

# 将拟合数据和原始数据（训练）拼接，计算差值并保存为3.csv
forecast_df = pd.DataFrame({'Fitted Data': fitted_values_lstm.flatten()}, index=train_data.index)
combined_data_3 = pd.concat([train_data, forecast_df], axis=1)
combined_data_3['Diff'] = combined_data_3['data_value'] - combined_data_3['Fitted Data']
combined_data_3.to_csv('3.csv')

# 将预测数据和原始数据（测试）拼接，计算差值并保存为4.csv
forecast_df = pd.DataFrame({'Forecast Data': forecasted_values_lstm.flatten()}, index=test_data.index)
combined_data_4 = pd.concat([test_data, forecast_df], axis=1)
combined_data_4['Diff'] = combined_data_4['data_value'] - combined_data_4['Forecast Data']
combined_data_4.to_csv('4.csv')

# Evaluation metrics for training set
train_predictions = fitted_values_lstm.flatten()
train_actual = train_data['data_value'].values
train_residuals = train_actual - train_predictions

train_mae = mean_absolute_error(train_actual, train_predictions)
train_mse = mean_squared_error(train_actual, train_predictions)
train_mape = np.mean(np.abs(train_residuals / train_actual)) * 100
train_mbe = np.mean(train_residuals)
train_r2 = r2_score(train_actual, train_predictions)

print("Training Set Metrics:")
print(f"MAE: {train_mae}")
print(f"MSE: {train_mse}")
print(f"MAPE: {train_mape}")
print(f"MBE: {train_mbe}")
print(f"R2: {train_r2}")

# Evaluation metrics for forecasted values
forecast_predictions = forecasted_values_lstm.flatten()
forecast_actual = test_data['data_value'].values
forecast_residuals = forecast_actual - forecast_predictions

forecast_mae = mean_absolute_error(forecast_actual, forecast_predictions)
forecast_mse = mean_squared_error(forecast_actual, forecast_predictions)
forecast_mape = np.mean(np.abs(forecast_residuals / forecast_actual)) * 100
forecast_mbe = np.mean(forecast_residuals)
forecast_r2 = r2_score(forecast_actual, forecast_predictions)

print("\nForecast Metrics:")
print(f"MAE: {forecast_mae}")
print(f"MSE: {forecast_mse}")
print(f"MAPE: {forecast_mape}")
print(f"MBE: {forecast_mbe}")
print(f"R2: {forecast_r2}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['data_value'], label='Original Data', color='#7FFFD4')
plt.plot(train_data.index, fitted_values_lstm, label='Fitted Data (LSTM Adjusted)', linestyle='--', color='#FFCAD4')
plt.plot(test_data.index, forecasted_values_lstm, label='Forecast Data (LSTM Adjusted)', linestyle='--', color='#F64E60')
plt.title('ARIMA(0,1,14) Model Adjusted with LSTM')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('ARIMA-LSTM结果.svg', format='svg', bbox_inches='tight')
plt.show()
