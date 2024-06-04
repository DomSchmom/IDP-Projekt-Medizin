import datetime
import os
import pandas as pd
import unisens
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ImuDataImport import import_imu_data, butter_lowpass_filter

# Load data
#data = pd.read_csv('~/Downloads/yc-p023/YC-P0230001_6D.tsv', delimiter='\t', skiprows=41)
#data = data[['Time', 'r_hand_rigid_body X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']]

imu_folder_path = '/Users/dominik/Downloads/imu'
accel_freq = 64
#import imu data and print the dataframes
data = import_imu_data(imu_folder_path, accel_freq)

numeric_columns = ['a_xl', 'a_yl', 'a_zl', 'a_xr', 'a_yr', 'a_zr']
imu_numeric_data = data[numeric_columns]
cutoff_freq = 2.0  # Adjust the cutoff frequency as needed
for col in numeric_columns:
    imu_numeric_data[col] = butter_lowpass_filter(imu_numeric_data[col], cutoff=cutoff_freq, fs=accel_freq)

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(imu_numeric_data.iloc[:, 1:])  # Exclude Time for scaling

# Split data into training and testing datasets
X_train = imu_numeric_data[:2 * 3600 * accel_freq]
X_test = imu_numeric_data[2 * 3600 * accel_freq:20 * 3600 * accel_freq]
time_test = data['timestamp'].iloc[2 * 3600 * accel_freq:20 * 3600 * accel_freq]

# Set up One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='auto')
oc_svm.fit(X_train)

# Calculate the decision function values
decision_function = oc_svm.decision_function(X_test)

# Plotting the time-error diagram
plt.figure(figsize=(12, 6))
plt.scatter(time_test, decision_function, color='blue')
#plt.plot(time_test, decision_function, label='Decision Function Value', color='red')
plt.axhline(y=0, color='green', linestyle='--', label='Decision Boundary')
plt.title('Time-Error Diagram for One-Class SVM')
plt.xlabel('Time')
plt.ylabel('Decision Function Value')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
