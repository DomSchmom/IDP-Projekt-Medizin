import pandas as pd
import matplotlib.pyplot as plt

left_wrist_file = 'csv/MP-PO002LeftWrist-timeSeries.csv.gz'
right_wrist_file = 'csv/MP-PO002RightWrist-timeSeries.csv.gz'
left_wrist_df = pd.read_csv(left_wrist_file, parse_dates=['time'])
right_wrist_df = pd.read_csv(right_wrist_file, parse_dates=['time'])

left_wrist_df['time'] = left_wrist_df['time'].str.replace(r'\s*\[UTC\]', '', regex=True)
left_wrist_df['time'] = pd.to_datetime(left_wrist_df['time'], utc=True).dt.tz_localize(None)
right_wrist_df['time'] = right_wrist_df['time'].str.replace(r'\s*\[UTC\]', '', regex=True)
right_wrist_df['time'] = pd.to_datetime(right_wrist_df['time'], utc=True).dt.tz_localize(None)
merged_df = pd.merge(left_wrist_df, right_wrist_df, on='time', suffixes=('_left', '_right'))

activity_columns = ['bicycling', 'mixed', 'sit-stand', 'sleep', 'vehicle', 'walking']

activity_results = {}
total_left_predictions = {activity: merged_df[f"{activity}_left"].sum() for activity in activity_columns}
total_right_predictions = {activity: merged_df[f"{activity}_right"].sum() for activity in activity_columns}
total_activities = 0
matching_activities = 0

for activity in activity_columns:

    left_column = f"{activity}_left"
    right_column = f"{activity}_right"
    matches = (merged_df[left_column] == 1) & (merged_df[right_column] == 1)
    total_for_activity = merged_df[left_column].sum()
    match_percentage = (matches.sum() / total_for_activity) * 100 if total_for_activity > 0 else 0
    activity_results[activity] = match_percentage
    matching_activities += matches.sum()
    total_activities += merged_df[left_column].sum()

general_match_percentage = (matching_activities / total_activities) * 100

fig, ax1 = plt.subplots(figsize=(12, 6))
activities = list(activity_results.keys())
percentages = list(activity_results.values())
left_totals = [total_left_predictions[act] for act in activities]
right_totals = [total_right_predictions[act] for act in activities]

ax1.bar(activities, percentages, color='skyblue', label='Specific Activity Match Percentage')
ax1.axhline(y=general_match_percentage, color='red', linestyle='--', label='Overall Match Percentage')
ax1.set_ylabel('Match Percentage (%)')
ax1.set_xlabel('Activities')
ax1.set_xticks(range(len(activities)))
ax1.set_xticklabels(activities, rotation=45)
ax1.set_title('Activity Match Percentage and Total Predictions')
ax2 = ax1.twinx()
ax2.plot(activities, left_totals, 'o-', color='green', label='Total Left Predictions')
ax2.plot(activities, right_totals, 'o-', color='purple', label='Total Right Predictions')
ax2.set_ylabel('Total Predictions')
ax1.spines['left'].set_position(('outward', 0))
ax2.spines['right'].set_position(('outward', 0))
ax1.set_ylim(0, max(percentages) + 10)
ax2.set_ylim(0, max(max(left_totals), max(right_totals)) + 10)
ax1.yaxis.set_label_position("left")
ax1.yaxis.tick_left()
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
ax1.grid(False)
ax2.grid(False)
fig.tight_layout()

plt.show()