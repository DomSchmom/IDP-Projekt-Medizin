from ImuDataImport import butter_lowpass_filter, import_imu_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load and preprocess data
imu_folder_path = '/Users/dominik/Downloads/imu'
accel_freq = 64
imu_data = import_imu_data(imu_folder_path, accel_freq)
pd.set_option('display.max_columns', None)

numeric_columns = ['a_xl', 'a_yl', 'a_zl', 'a_xr', 'a_yr', 'a_zr']
imu_numeric_data = imu_data[numeric_columns]

cutoff_freq = 1.5  # Adjust the cutoff frequency as needed
for col in numeric_columns:
    imu_numeric_data[col] = butter_lowpass_filter(imu_numeric_data[col], cutoff=cutoff_freq, fs=accel_freq)
timestamps = imu_data['timestamp']
timestamps = timestamps[2 * 3600 * accel_freq:24 * 3600 * accel_freq]

X_train = imu_numeric_data[:2*3600 * accel_freq]  # 10 seconds of data
X_test = imu_numeric_data[2*3600 * accel_freq:24 * 3600 * accel_freq]  # Another 10 seconds of data

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the autoencoder model
input_dim = X_train_scaled.shape[1]
encoding_dim = 14  # Number of neurons in the encoding layer

autoencoder = MLPRegressor(hidden_layer_sizes=(encoding_dim,), activation='relu', solver='adam', max_iter=200, random_state=42)

# Train the autoencoder
autoencoder.fit(X_train_scaled, X_train_scaled)

# Use the trained autoencoder to reconstruct the test data
X_test_pred = autoencoder.predict(X_test_scaled)
reconstruction_errors = np.mean(np.abs(X_test_scaled - X_test_pred), axis=1)

# Plot reconstruction errors
plt.figure(figsize=(12, 6))
plt.plot(timestamps, reconstruction_errors, label='Reconstruction error')
plt.axhline(y=np.mean(reconstruction_errors) + 2*np.std(reconstruction_errors), color='r', linestyle='--', label='Anomaly threshold')
plt.title('Reconstruction Error')
plt.xlabel('Time')
plt.ylabel('Reconstruction error')
plt.legend()
plt.show()

# Highlight the anomalies
anomaly_threshold = np.mean(reconstruction_errors) + 2*np.std(reconstruction_errors)
anomalies = reconstruction_errors > anomaly_threshold

plt.figure(figsize=(12, 6))
plt.plot(timestamps, reconstruction_errors, label='Reconstruction error')
plt.fill_between(timestamps, 0, 1, where=anomalies, color='red', alpha=0.5, transform=plt.gca().get_xaxis_transform(), label='Anomalies')
plt.axhline(y=anomaly_threshold, color='r', linestyle='--', label='Anomaly threshold')
plt.title('Reconstruction Error with Anomalies')
plt.xlabel('Time')
plt.ylabel('Reconstruction error')
plt.legend()
plt.show()
