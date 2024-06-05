from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np

from ImuDataImport import butter_lowpass_filter, import_imu_data


# Function to plot the frequency spectrum
def plot_frequency_spectrum(data, sampling_rate):
    data = np.asarray(data)
    N = len(data)
    T = 1.0 / sampling_rate
    yf = fft(data)
    xf = fftfreq(N, T)[:N // 2]
    plt.figure(figsize=(12, 6))
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

imu_folder_path = '/Users/dominik/Downloads/imu'
accel_freq = 64
imu_data = import_imu_data(imu_folder_path, accel_freq)
timestamps = imu_data['timestamp']
imu_data = imu_data['a_xl']  # Use only one column for demonstration purposes


# Plot the original frequency spectrum
plot_frequency_spectrum(imu_data, accel_freq)

# Try different cutoff frequencies and observe the effect
cutoff_freqs = [5,3,1,0.5]  # Example cutoff frequencies to try
filtered_data = {}

for cutoff in cutoff_freqs:
    filtered_data[cutoff] = butter_lowpass_filter(imu_data, cutoff=cutoff, fs=accel_freq)

# Plot the filtered data for different cutoff frequencies
plt.figure(figsize=(12, 6))
plt.plot(timestamps, imu_data, label='Original Data', alpha=0.5)

for cutoff in cutoff_freqs:
    plt.plot(timestamps, filtered_data[cutoff], label=f'Cutoff = {cutoff}')

plt.legend()
plt.title("Effect of Different Cutoffs on IMU Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
