import torch

from ImuDataImport import butter_lowpass_filter, import_imu_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import plot_acf, plot_residuals_analysis

# Load and preprocess data
imu_folder_path = '/Users/dominik/Downloads/imu'
accel_freq = 64
imu_data = import_imu_data(imu_folder_path, accel_freq)
imu_data = imu_data[:3 * 3600 * accel_freq]  # Use only a subset of the data

numeric_columns = ['a_xl', 'a_yl', 'a_zl', 'a_xr', 'a_yr', 'a_zr']
imu_numeric_data = imu_data[numeric_columns]

cutoff_freq = 2  # Adjust the cutoff frequency as needed
for col in numeric_columns:
    imu_numeric_data[col] = butter_lowpass_filter(imu_numeric_data[col], cutoff=cutoff_freq, fs=accel_freq)
timestamps = imu_data['timestamp']

# Convert to Darts TimeSeries
imu_series = TimeSeries.from_dataframe(imu_data, 'timestamp', numeric_columns)

# Normalize the data
scaler = Scaler()
imu_series_scaled = scaler.fit_transform(imu_series)
imu_series_scaled = imu_series_scaled.astype(np.float32)

# Split the data into training and test sets
train, test = imu_series_scaled.split_before(2 * 3600 * accel_freq)
timestamps = timestamps[-len(test):]

# Define the autoencoder model
model = RNNModel(
    model='LSTM',
    input_chunk_length=30,
    output_chunk_length=30,
    hidden_dim=20,
    n_rnn_layers=1,
    dropout=0.1,
    batch_size=32,
    n_epochs=2,
    random_state=42,
    training_length=30
)

# Train the autoencoder
model.fit(train)

# Predict the test data
pred = model.predict(len(test))

# Compute reconstruction errors
reconstruction_errors = np.mean(np.abs(test.values() - pred.values()), axis=1)

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
