
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from ImuDataImport import import_imu_data, butter_lowpass_filter
import pandas as pd
import numpy as np

def create_acceleration_features(data, window_size, threshold=1.1):
    """
    Create features to capture strong accelerations lasting over 1 second in one direction.
    """
    window_size = accel_freq  # 1 second window
    acc_columns = ['a_xl', 'a_yl', 'a_zl', 'a_xr', 'a_yr', 'a_zr']
    sustained_features = pd.DataFrame()

    for col in acc_columns:
        # Calculate rolling mean and standard deviation
        rolling_mean = data[col].rolling(window=window_size).mean()
        rolling_std = data[col].rolling(window=window_size).std()

        # Identify strong sustained accelerations
        sustained_acc = (data[col].abs() > threshold) & (rolling_std < threshold)
        sustained_acc = sustained_acc.astype(int).rolling(window=window_size).sum()

        # Create a binary feature for sustained acceleration over 1 second
        sustained_features[col + '_sustained'] = (sustained_acc >= window_size).astype(int)

    sustained_features = sustained_features.dropna()
    return sustained_features

def calculate_velocity(data, accel_freq):
    """
    Calculate velocity from acceleration data.
    """
    acc_columns = ['a_xl', 'a_yl', 'a_zl', 'a_xr', 'a_yr', 'a_zr']
    velocity_data = pd.DataFrame()

    for col in acc_columns:
        velocity_data[col.replace('a', 'v')] = data[col].cumsum() / accel_freq

    cutoff_freq = 10  # Adjust the cutoff frequency as needed
    for col in velocity_data:
        velocity_data[col] = butter_lowpass_filter(velocity_data[col], cutoff=cutoff_freq, fs=accel_freq)

    return velocity_data

def create_window_features(data, window_size):
    """
    Create rolling window features for the dataset.
    """
    window_features = data.rolling(window=window_size).mean().dropna()
    return window_features

accel_freq = 64
imu_data = import_imu_data('/Users/dominik/Downloads/imu', accel_freq)

numeric_columns = ['a_xl', 'a_yl', 'a_zl', 'a_xr', 'a_yr', 'a_zr']
imu_numeric_data = imu_data[numeric_columns]

velocity_data = calculate_velocity(imu_numeric_data, accel_freq)
timestamps = imu_data['timestamp']

X_train = imu_numeric_data[:2 * 3600 * accel_freq]
X_test = imu_numeric_data[2 * 3600 * accel_freq:23 * 3600 * accel_freq]
timestamps_test_raw = timestamps[2 * 3600 * accel_freq:23 * 3600 * accel_freq]


# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

timestamps_test = timestamps_test_raw[len(timestamps_test_raw) - len(X_test):]

# Train the Isolation Forest model
isolation_forest = IsolationForest(contamination=0.001, random_state=42)
isolation_forest.fit(X_train)

# Predict anomalies on the test set
test_scores = isolation_forest.decision_function(X_test)
anomalies = isolation_forest.predict(X_test)

# Plot the anomalies
plt.figure(figsize=(10, 6))
plt.plot(timestamps_test, test_scores, label='Isolation Forest Scores')
plt.scatter(timestamps_test[anomalies == -1], test_scores[anomalies == -1], color='red', label='Anomalies')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Isolation Forest Scores')
plt.title('Anomaly Detection with Isolation Forest')
plt.show()

# Count and print the number of anomalies detected
num_anomalies = np.sum(anomalies == -1)
print(f'Number of anomalies detected: {num_anomalies}')
