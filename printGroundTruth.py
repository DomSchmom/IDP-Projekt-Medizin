import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ground_truth_df = pd.read_csv('csv/DominikLabeledIMU.csv')
ground_truth_df['from'] = pd.to_datetime(ground_truth_df['from'], format='%d-%b-%Y %I:%M:%S %p').dt.tz_localize(None)

ground_truth_df['duration'] = pd.to_timedelta(ground_truth_df['duration'])
ground_truth_df['to'] = ground_truth_df['from'] + ground_truth_df['duration']

activity_colors = {
    'sleep': 'navy',
    'sit-stand': 'red',
    'walking': 'lime',
    'mixed': 'green',
    'bicycling': 'cyan',
    'vehicle': 'brown'
}

ground_truth_df['day'] = ground_truth_df['from'].dt.date

days = ground_truth_df['day'].unique()
fig, axes = plt.subplots(len(days), 1, figsize=(12, 2 * len(days)))  # Make the plot 1.5 times longer

if len(days) == 1:
    axes = [axes]

for i, (ax, day) in enumerate(zip(axes, days)):
    day_data = ground_truth_df[ground_truth_df['day'] == day]

    if i > 0:
        prev_day_data = ground_truth_df[ground_truth_df['day'] == days[i - 1]]
        overlap_data = prev_day_data[prev_day_data['to'] > pd.Timestamp(f'{day} 00:00:00')]
        day_data = pd.concat([overlap_data, day_data])

    for _, row in day_data.iterrows():
        start_time = row['from']
        end_time = row['to']
        activity = row['type']
        color = activity_colors.get(activity, 'grey')
        ax.hlines(y=1, xmin=start_time, xmax=end_time, color=color, linewidth=50)  # Increased linewidth

    for hour in range(0, 24, 1):
        ax.axvline(pd.Timestamp(f'{day} {hour:02}:00:00'), color='lightgrey', linestyle='--')

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlim(pd.Timestamp(f'{day} 00:00:00'), pd.Timestamp(f'{day} 23:59:59'))
    ax.set_title(f'Ground Truth Activities for {day}')
    ax.set_yticks([])
    ax.set_yticklabels([])

legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for activity, color in activity_colors.items()]
legend_labels = list(activity_colors.keys())
fig.legend(legend_handles, legend_labels, loc='upper left')

for ax in axes[:-1]:
    ax.set_xticklabels([])
axes[-1].set_xlabel('Time')

plt.tight_layout()
plt.show()