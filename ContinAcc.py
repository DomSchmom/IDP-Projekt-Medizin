import os
import pandas as pd
import datetime
import unisens
import matplotlib.pyplot as plt
from ImuDataImport import import_imu_data, butter_lowpass_filter


def detect_sustained_accelerations(data, threshold, window_size):
    """
    Detect sustained accelerations over a given threshold for more than a specified window size.
    """
    sustained_periods = (data.abs() > threshold).rolling(window=window_size).sum() >= window_size
    return sustained_periods

# Import IMU data
accel_freq = 64
imu_data = import_imu_data('/Users/dominik/Downloads/imu', accel_freq)

# Define numeric columns and timestamps
acc_columns = ['a_xl', 'a_yl', 'a_zl', 'a_xr', 'a_yr', 'a_zr']
timestamps = imu_data['timestamp']
timestamps = timestamps[:24 * 3600 * accel_freq]

# Use the original data without downsampling
imu_numeric_data = imu_data[acc_columns]
#only take the first two days of data
imu_numeric_data = imu_numeric_data[:24 * 3600 * accel_freq]
cutoff_freq = 1  # Adjust the cutoff frequency as needed
for col in imu_numeric_data:
    imu_numeric_data[col] = butter_lowpass_filter(imu_numeric_data[col], cutoff=cutoff_freq, fs=accel_freq)
# Detect sustained accelerations
threshold = 1  # Adjust the threshold as needed
window_size = accel_freq * 10  # 1-second window

sustained_accel = pd.DataFrame()
for col in acc_columns:
    sustained_accel[col] = detect_sustained_accelerations(imu_numeric_data[col], threshold, window_size)

# Combine sustained accelerations
combined_sustained_accel = sustained_accel.any(axis=1)

# Plot the accelerations
plt.figure(figsize=(15, 10))
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for i, col in enumerate(acc_columns):
    plt.plot(timestamps, imu_numeric_data[col], color=colors[i], label=f'{col} acceleration')

# Highlight sustained accelerations with shaded regions
combined_sustained_accel = combined_sustained_accel.reset_index(drop=True)
start = None
for i in range(len(combined_sustained_accel)):
    if combined_sustained_accel[i] and start is None:
        start = timestamps.iloc[i]
    elif not combined_sustained_accel[i] and start is not None:
        plt.axvspan(start, timestamps.iloc[i-1], color='red', alpha=0.3)
        start = None

if start is not None:
    plt.axvspan(start, timestamps.iloc[-1], color='red', alpha=0.3)

plt.legend(loc='upper right')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.title('Sustained Accelerations')
plt.show()