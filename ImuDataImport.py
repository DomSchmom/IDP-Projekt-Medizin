import os
import pandas as pd
import datetime
import unisens
from scipy.signal import butter, filtfilt


def import_imu_data(imu_folder_path, accel_freq):
    print('IMU Data is being imported...')
    imu_data = pd.DataFrame()
    imu_data_r = pd.DataFrame()
    imu_data_l = pd.DataFrame()

    for root, dirs, files in os.walk(imu_folder_path):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(root, file)
                unisens_directory = os.path.dirname(xml_file_path)
                u = unisens.Unisens(unisens_directory)
                metadata = u['customAttributes']
                raw_accel_data = u['acc.bin'].get_data()
                raw_gyro_data = u['angularrate.bin'].get_data()
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

                if sensorLocation == 'right_wrist':
                    imu_data_r['a_xr'] = pd.DataFrame(raw_accel_data[0])
                    imu_data_r['a_yr'] = pd.DataFrame(raw_accel_data[1])
                    imu_data_r['a_zr'] = pd.DataFrame(raw_accel_data[2])
                    imu_data_r['g_xr'] = pd.DataFrame(raw_gyro_data[0])
                    imu_data_r['g_yr'] = pd.DataFrame(raw_gyro_data[1])
                    imu_data_r['g_zr'] = pd.DataFrame(raw_gyro_data[2])
                    imu_data_r['timestamp'] = pd.date_range(start=imu_timestampStartUTC, periods=len(imu_data_r),
                                                            freq=pd.Timedelta(seconds=delta_time))

                elif sensorLocation == 'left_wrist':
                    imu_data_l['a_xl'] = pd.DataFrame(raw_accel_data[0])
                    imu_data_l['a_yl'] = pd.DataFrame(raw_accel_data[1])
                    imu_data_l['a_zl'] = pd.DataFrame(raw_accel_data[2])
                    imu_data_l['g_xl'] = pd.DataFrame(raw_gyro_data[0])
                    imu_data_l['g_yl'] = pd.DataFrame(raw_gyro_data[1])
                    imu_data_l['g_zl'] = pd.DataFrame(raw_gyro_data[2])
                    imu_data_l['timestamp'] = pd.date_range(start=imu_timestampStartUTC, periods=len(imu_data_l),
                                                            freq=pd.Timedelta(seconds=delta_time))

    imu_data = pd.merge_asof(imu_data_l, imu_data_r, on='timestamp', direction='nearest', tolerance=pd.Timedelta('15ms'))
    imu_data.sort_values(by='timestamp', inplace=True)
    imu_data.reset_index(drop=True, inplace=True)
    imu_data = imu_data[3 * 60 * accel_freq:-3 * 60 * accel_freq]
    print('IMU Data DF has been created with both right and left hand data...')
    return imu_data

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)