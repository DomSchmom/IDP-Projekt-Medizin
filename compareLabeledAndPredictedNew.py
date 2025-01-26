import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import timedelta

labeled_df = pd.read_csv('csv\\myDataWithAnnotations.csv', parse_dates=['time'])
predictions_df = pd.read_csv('results\\mydata-timeSeries.csv', parse_dates=['time'])

labeled_df['time'] = pd.to_datetime(labeled_df['time'])
predictions_df['time'] = predictions_df['time'].str.replace(r'\s*\[UTC\]', '', regex=True)
predictions_df['time'] = pd.to_datetime(predictions_df['time'], utc=True)
predictions_df['time'] = predictions_df['time'].dt.tz_localize(None)

activity_columns = ['bicycling', 'mixed', 'sit-stand', 'sleep', 'vehicle', 'walking']

def get_predicted_activity(row):
    activity_values = row[activity_columns]
    max_value = activity_values.max()
    if max_value == 0:
        return 'unknown'
    else:
        predicted_activity = activity_values.idxmax()
        return predicted_activity

predictions_df['predicted_activity'] = predictions_df.apply(get_predicted_activity, axis=1)

labeled_df.set_index('time', inplace=True)
predictions_df.set_index('time', inplace=True)

labeled_df.sort_index(inplace=True)
predictions_df.sort_index(inplace=True)

print(f"Total windows in labeled_df: {len(labeled_df)}")
print(f"Total windows in predictions_df: {len(predictions_df)}")
def get_true_label(pred_time):
    time_window = timedelta(seconds=10)
    start_time = pred_time - time_window
    end_time = pred_time + time_window
    annotations_in_window = labeled_df.loc[start_time:end_time, 'annotation']
    if annotations_in_window.empty:
        return 'unknown'
    else:
        return annotations_in_window.mode().iloc[0]

predictions_df['true_label'] = predictions_df.index.to_series().apply(get_true_label)
print(f"Total windows before filtering unknowns: {len(predictions_df)}")

valid_rows = predictions_df[(predictions_df['true_label'] != 'unknown') & (predictions_df['predicted_activity'] != 'unknown')]
true_labels = valid_rows['true_label']
predicted_labels = valid_rows['predicted_activity']

print('Classifier performance on test data:')
print(classification_report(true_labels, predicted_labels, zero_division=0))

classes = sorted(set(true_labels.unique()) | set(predicted_labels.unique()))

cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()