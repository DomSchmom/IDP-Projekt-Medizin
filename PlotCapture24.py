import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:\\Users\\Domi9\\Downloads\\capture24\\capture24\\P001.csv\\P001.csv'

df = pd.read_csv(file_path)
df = df[['time', 'annotation']]
df['time'] = pd.to_datetime(df['time'])
df['annotation'] = df['annotation'].astype(str)

df = df.iloc[::1000, :]

for index, row in df.iterrows():
    print(f"Timestamp: {row['time']}, Annotation: {row['annotation']}")