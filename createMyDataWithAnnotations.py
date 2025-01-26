import pandas as pd

labels_df = pd.read_csv('csv\\DominikLabeledIMU.csv')
imu_df = pd.read_csv('csv\\myData.csv')

labels_df['from'] = pd.to_datetime(labels_df['from'])
labels_df['to'] = pd.to_datetime(labels_df['to'])
imu_df['timestamp'] = pd.to_datetime(imu_df['timestamp'])

imu_df['annotation'] = None

for index, row in labels_df.iterrows():
    start_time = row['from']
    end_time = row['to']
    annotation = row['type']

    mask = (imu_df['timestamp'] >= start_time) & (imu_df['timestamp'] <= end_time)
    imu_df.loc[mask, 'annotation'] = annotation

imu_df = imu_df[imu_df['annotation'].notna()]

imu_df = imu_df.rename(columns={
    'timestamp': 'time',
    'a_xr': 'x',
    'a_yr': 'y',
    'a_zr': 'z'
})
imu_df = imu_df[['time', 'x', 'y', 'z', 'annotation']]
imu_df.to_csv('myDataWithAnnotations.csv', index=False)
