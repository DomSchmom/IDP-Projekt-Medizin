import os

import numpy as np
import pandas as pd
import datetime
import unisens
from scipy.signal import butter, filtfilt


def saveToCSV(imu_folder_path, accel_freq):
    print('IMU Data is being imported...')
    imu_data = pd.DataFrame()
    imu_data_r = pd.DataFrame()

    for root, dirs, files in os.walk(imu_folder_path):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(root, file)
                unisens_directory = os.path.dirname(xml_file_path)
                u = unisens.Unisens(unisens_directory)
                metadata = u['customAttributes']
                raw_accel_data = u['acc.bin'].get_data()
                temperature_data = u['temp.bin'].get_data()
                metadata_dict = vars(metadata)
                sensorTimeDrift = metadata_dict['attrib']['sensorTimeDrift']
                sensorLocation = metadata_dict['attrib']['sensorLocation']
                gender = metadata_dict['attrib']['gender']
                personId = metadata_dict['attrib']['personId']
                imu_timestampStartUTC = pd.to_datetime(metadata_dict['attrib']['timestampStartUTC'],
                                                       format='%Y-%m-%dT%H:%M:%S.%f')
                timezone_offset_seconds = int(metadata_dict['attrib']['timeZoneOffset'])
                imu_timestampStartUTC += datetime.timedelta(seconds=timezone_offset_seconds)
                imu_timestampStartUTC = imu_timestampStartUTC.strftime('%Y-%m-%d %H:%M:%S')
                freq = 1 / accel_freq
                delta_time = round(freq, 4)

                if sensorLocation == 'left_wrist':
                    imu_data_r['a_xr'] = pd.DataFrame(raw_accel_data[0])
                    imu_data_r['a_yr'] = pd.DataFrame(raw_accel_data[1])
                    imu_data_r['a_zr'] = pd.DataFrame(raw_accel_data[2])
                    imu_data_r['timestamp'] = pd.date_range(start=imu_timestampStartUTC, periods=len(imu_data_r),
                                                            freq=pd.Timedelta(seconds=delta_time))
                    temperature_df = pd.DataFrame(temperature_data[0], columns=['temperature'])

                    repeated_temperature = np.repeat(temperature_df['temperature'].values, 64)
                    repeated_temperature_df = pd.DataFrame(repeated_temperature, columns=['temperature'])

                    imu_data_r['temperature'] = repeated_temperature_df['temperature']



    imu_data = imu_data_r
    imu_data.reset_index(drop=True, inplace=True)
    imu_data = imu_data[3 * 60 * accel_freq: 48 * 60 * 60 * accel_freq]
    csv_file_path = "csv\\MP-PO002LeftWrist.csv"

    imu_data.to_csv(csv_file_path, index=False)
    print('IMU Data DF has been created...')

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

saveToCSV("F:\\paulavillafulton\\eXprt-backup\\05022024\\data\\MP-P002\\imu\\2023-11-10 10.02.26_result_2023-11-30 09.39.57", 64)