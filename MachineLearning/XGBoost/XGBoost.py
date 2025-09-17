# model1 时间序列分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the CSV data
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\1.csv"
df = pd.read_csv(file_path)

# Assuming 'data' is the column containing your data
X = df['data'].values
y = np.roll(X, -1)  # Shift the target values by one position for predicting the next value

# Drop the last value in X and y to align the lengths
X = X[:-1]
y = y[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape the data to be a 2D array
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Create and train the XGBoost model
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Make predictions for the test set
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Save the predictions for all y values in '2.csv'
df_all_predictions = pd.DataFrame({'True Values': y, 'Predicted Values': model.predict(X.reshape(-1, 1))})
df_all_predictions.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\2.csv", index=False)

# Make predictions for future values
known_value = df['data'].iloc[-1]
n = 100  # Number of future values to predict
predicted_values = []

# Make predictions for the next n values
for i in range(n):
    next_value = model.predict(np.array([[known_value]]))
    predicted_values.append(next_value[0])
    known_value = next_value[0]  # Update the known value for the next iteration

# Save the predictions for future values in '3.csv' with 16 decimal places
df_future_predictions = pd.DataFrame({'Predicted Values': predicted_values})
df_future_predictions['Predicted Values'] = df_future_predictions['Predicted Values'].apply(lambda x: f'{x:.16f}')
df_future_predictions.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\3.csv", index=False)

# Calculate RMSE for the training set
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f'Training Set RMSE: {rmse_train}')

# Calculate RMSE for the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f'Test Set RMSE: {rmse_test}')

# Calculate MAE, MBE, and R2 for the training set
mae_train = mean_absolute_error(y_train, y_pred_train)
mbe_train = np.mean(y_pred_train - y_train)
r2_train = r2_score(y_train, y_pred_train)

print(f'Training Set MAE: {mae_train:.8f}')
print(f'Training Set MBE: {mbe_train:.8f}')
print(f'Training Set R2: {r2_train:.8f}')

# Calculate MAE, MBE, and R2 for the test set
mae_test = mean_absolute_error(y_test, y_pred_test)
mbe_test = np.mean(y_pred_test - y_test)
r2_test = r2_score(y_test, y_pred_test)

print(f'Test Set MAE: {mae_test:.8f}')
print(f'Test Set MBE: {mbe_test:.8f}')
print(f'Test Set R2: {r2_test:.8f}')

# Visualize the results with custom colors
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='#B2B2FF', label='Actual vs. Predicted (Training Set)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='#FFC6CC', linewidth=2, label='Perfect Prediction')
plt.title(f'XGBoost Regression - Training Set\nRMSE: {rmse_train:.8f}\nMAE: {mae_train:.8f}\nR2: {r2_train:.8f}\nMBE: {mbe_train:.8f}')
plt.xlabel('Actual Values (Training Set)')
plt.ylabel('Predicted Values (Training Set)')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='#B2B2FF', label='Actual vs. Predicted (Test Set)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='#FFC6CC', linewidth=2, label='Perfect Prediction')
plt.title(f'XGBoost Regression - Test Set\nRMSE: {rmse_test:.8f}\nMAE: {mae_test:.8f}\nR2: {r2_test:.8f}\nMBE: {mbe_test:.8f}')
plt.xlabel('Actual Values (Test Set)')
plt.ylabel('Predicted Values (Test Set)')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\scatter-scatter1.svg")
plt.show()
plt.close()

# Visualize the results with custom colors
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.plot(y_train, label='True Values (Training Set)', color='#B2B2FF')
plt.scatter(range(len(y_train)), y_pred_train, label='Predicted Values (Training Set)', color='#FFC6CC', marker='o')
plt.title(f'XGBoost Regression - Training Set\nTraining Set RMSE: {rmse_train:.8f}')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True Values (Test Set)', color='#B2B2FF')
plt.scatter(range(len(y_test)), y_pred_test, label='Predicted Values (Test Set)', color='#FFC6CC', marker='o')
plt.title(f'XGBoost Regression - Test Set\nTest Set RMSE: {rmse_test:.8f}')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\scatter-line1.svg")
plt.show()
plt.close()

# Visualize the results with custom colors
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.plot(y_train, label='True Values (Training Set)', color='#B2B2FF')
plt.plot(y_pred_train, label='Predicted Values (Training Set)', color='#FFC6CC')
plt.title(f'XGBoost Regression - Training Set\nTraining Set RMSE: {rmse_train:.8f}')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True Values (Test Set)', color='#B2B2FF')
plt.plot(y_pred_test, label='Predicted Values (Test Set)', color='#FFC6CC')
plt.title(f'XGBoost Regression - Test Set\nTest Set RMSE: {rmse_test:.8f}')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\line-line1.svg")
plt.show()
plt.close()



# model2 回归分析
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\1.1.csv"
df = pd.read_csv(file_path)

# Separate the features (independent variables) and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Make predictions for the test set
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Save the predictions for all y values in '1.2.csv'
df_all_predictions = pd.DataFrame({'True Values': y, 'Predicted Values': model.predict(X)})
df_all_predictions.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\1.2.csv", index=False)

# Make predictions for future values
known_value = X[-1]  # Using the last row of features as the known value
n = 100  # Number of future values to predict
predicted_values = []

# Make predictions for the next n values
for i in range(n):
    next_value = model.predict(np.array([known_value]))
    predicted_values.append(next_value[0])
    known_value = np.append(known_value[1:], next_value[0])  # Update the known value for the next iteration

# Save the predictions for future values in '1.3.csv' with 16 decimal places
df_future_predictions = pd.DataFrame({'Predicted Values': predicted_values})
df_future_predictions['Predicted Values'] = df_future_predictions['Predicted Values'].apply(lambda x: f'{x:.16f}')
df_future_predictions.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\1.3.csv", index=False)

# Calculate metrics for training set
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
mbe_train = np.mean(y_pred_train - y_train)
r2_train = r2_score(y_train, y_pred_train)


# Calculate metrics for test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mbe_test = np.mean(y_pred_test - y_test)
r2_test = r2_score(y_test, y_pred_test)

# Print the results
print(f'Training Set RMSE: {rmse_train:.8f}')
print(f'Training Set MAE: {mae_train:.8f}')
print(f'Training Set MBE: {mbe_train:.8f}')
print(f'Training Set R2: {r2_train:.8f}')

print(f'Test Set RMSE: {rmse_test:.8f}')
print(f'Test Set MAE: {mae_test:.8f}')
print(f'Test Set MBE: {mbe_test:.8f}')
print(f'Test Set R2: {r2_test:.8f}')

# Visualize the results
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='#A5DEE4', label='Actual vs. Predicted (Training Set)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='#F9A825', linewidth=2, label='Perfect Prediction')
plt.title(f'XGBoost Regression - Training Set\nRMSE: {rmse_train:.8f}\nMAE: {mae_train:.8f}\nR2: {r2_train:.8f}\nMBE: {mbe_train:.8f}')
plt.xlabel('Actual Values (Training Set)')
plt.ylabel('Predicted Values (Training Set)')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='#A5DEE4', label='Actual vs. Predicted (Test Set)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='#F9A825', linewidth=2, label='Perfect Prediction')
plt.title(f'XGBoost Regression - Test Set\nRMSE: {rmse_test:.8f}\nMAE: {mae_test:.8f}\nR2: {r2_test:.8f}\nMBE: {mbe_test:.8f}')
plt.xlabel('Actual Values (Test Set)')
plt.ylabel('Predicted Values (Test Set)')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\scatter-scatter1.1.svg")
plt.show()
plt.close()

# Visualize the results with custom colors
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.plot(y_train, label='True Values (Training Set)', color='#A5DEE4')
plt.scatter(range(len(y_train)), y_pred_train, label='Predicted Values (Training Set)', color='#F9A825', marker='o')
plt.title(f'XGBoost Regression - Training Set\nTraining Set RMSE: {rmse_train:.8f}')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True Values (Test Set)', color='#A5DEE4')
plt.scatter(range(len(y_test)), y_pred_test, label='Predicted Values (Test Set)', color='#F9A825', marker='o')
plt.title(f'XGBoost Regression - Test Set\nTest Set RMSE: {rmse_test:.8f}')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\scatter-line1.1.svg")
plt.show()
plt.close()

# Visualize the results with custom colors
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.plot(y_train, label='True Values (Training Set)', color='#A5DEE4')
plt.plot(y_pred_train, label='Predicted Values (Training Set)', color='#F9A825')
plt.title(f'XGBoost Regression - Training Set\nTraining Set RMSE: {rmse_train:.8f}')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True Values (Test Set)', color='#A5DEE4')
plt.plot(y_pred_test, label='Predicted Values (Test Set)', color='#F9A825')
plt.title(f'XGBoost Regression - Test Set\nTest Set RMSE: {rmse_test:.8f}')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\line-line1.1.svg")
plt.show()
plt.close()



# model3 分类分析
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the CSV data with a specific encoding (e.g., 'latin1')
file_path = r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\1.1.1.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Separate the features (independent variables) and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Adjust class labels to start from 0
y = y - 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model for classification
model = XGBClassifier(objective='multi:softmax', num_class=len(set(y)))
model.fit(X_train, y_train)

# Save the predictions for all y values in '1.1.2.csv'
df_all_predictions = pd.DataFrame({'True Values': y, 'Predicted Values': model.predict(X)})
df_all_predictions.to_csv(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\1.1.2.csv", index=False)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics for training set
accuracy_train = accuracy_score(y_train, y_pred_train)
classification_report_train = classification_report(y_train, y_pred_train)
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

# Calculate metrics for the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
classification_report_test = classification_report(y_test, y_pred_test)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

# Print the results
print(f'Training Set Accuracy: {accuracy_train:.8f}')
print('Training Set Classification Report:')
print(classification_report_train)
print('Training Set Confusion Matrix:')
print(confusion_matrix_train)

print(f'\nTest Set Accuracy: {accuracy_test:.8f}')
print('Test Set Classification Report:')
print(classification_report_test)
print('Test Set Confusion Matrix:')
print(confusion_matrix_test)

# Define custom colors
custom_colors = ['#DCC6E0', '#F8E6FF', '#FFF0F5', '#A8DADC', '#A8DADC', '#70C1B3']

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list('custom', custom_colors, N=256)

# Visualize the confusion matrix without Seaborn
plt.figure(figsize=(12, 6))

# Plotting for the training set confusion matrix
plt.subplot(1, 2, 1)
plt.imshow(confusion_matrix_train, interpolation='nearest', cmap=cmap)
plt.title('Confusion Matrix - Training Set')
plt.colorbar()
classes_train = set(y_train)
tick_marks_train = range(len(classes_train))
plt.xticks(tick_marks_train, classes_train)
plt.yticks(tick_marks_train, classes_train)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Plotting for the test set confusion matrix
plt.subplot(1, 2, 2)
plt.imshow(confusion_matrix_test, interpolation='nearest', cmap=cmap)
plt.title('Confusion Matrix - Test Set')
plt.colorbar()
classes_test = set(y_test)
tick_marks_test = range(len(classes_test))
plt.xticks(tick_marks_test, classes_test)
plt.yticks(tick_marks_test, classes_test)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\Confusion Matrix1.1.1.svg")
plt.show()
plt.close()

# Visualize the results with custom colors
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.plot(y_train, label='True Values (Training Set)', color='#f99f9f')
plt.scatter(range(len(y_train)), y_pred_train, label='Predicted Values (Training Set)', color='#FFC107', marker='o')
plt.title('XGBoost Classification - Training Set')
plt.xlabel('Sample Index')
plt.ylabel('Class Labels')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True Values (Test Set)', color='#f99f9f')
plt.scatter(range(len(y_test)), y_pred_test, label='Predicted Values (Test Set)', color='#FFC107', marker='o')
plt.title('XGBoost Classification - Test Set')
plt.xlabel('Sample Index')
plt.ylabel('Class Labels')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\scatter-line1.1.1.svg")
plt.show()
plt.close()

# Visualize the results with custom colors
plt.figure(figsize=(12, 6))

# Plotting for the training set
plt.subplot(1, 2, 1)
plt.plot(y_train, label='True Values (Training Set)', color='#f99f9f')
plt.plot(y_pred_train, label='Predicted Values (Training Set)', color='#FFC107')
plt.title('XGBoost Classification - Training Set')
plt.xlabel('Sample Index')
plt.ylabel('Class Labels')
plt.legend()

# Plotting for the test set
plt.subplot(1, 2, 2)
plt.plot(y_test, label='True Values (Test Set)', color='#f99f9f')
plt.plot(y_pred_test, label='Predicted Values (Test Set)', color='#FFC107')
plt.title('XGBoost Classification - Test Set')
plt.xlabel('Sample Index')
plt.ylabel('Class Labels')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\23991\OneDrive\桌面\Python\venv\shuxuejianmo\shu_xue_jian_mo\XGBoost\line-line1.1.1.svg")
plt.show()
plt.close()
