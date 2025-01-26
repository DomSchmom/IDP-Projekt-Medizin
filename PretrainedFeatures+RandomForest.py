import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt

repo = 'OxWearables/ssl-wearables'
harnet10 = torch.hub.load(repo, 'harnet10', class_num=6, pretrained=True)

feature_extractor = nn.Sequential(*list(harnet10.children())[:-1])
feature_extractor.eval()

def extract_windows(data, winsize='10s'):
    X_windows, Y_labels = [], []
    for t, w in data.resample(winsize, origin='start'):
        if w.isna().any().any():
            continue
        if len(w) == 642:
            w = w.iloc[:641]
        elif len(w) != 641:
            continue
        x = w[['x', 'y', 'z']].to_numpy().T  # Transpose to (channels, sequence_length)
        y = w['annotation'].mode(dropna=False).item()
        X_windows.append(x)
        Y_labels.append(y)
    X_windows = np.stack(X_windows)
    Y_labels = np.array(Y_labels)
    return X_windows, Y_labels

data = pd.read_csv('csv/myDataWithAnnotations.csv', index_col='time', parse_dates=['time'])
X_data, Y_labels = extract_windows(data)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
)

def extract_features_batch(model, data, batch_size=128):
    features_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            print(f'Extracting features batch {i+1}/{len(data)}')
            x_batch = data[i:i+batch_size]
            x_tensor = torch.FloatTensor(x_batch)
            feature = model(x_tensor)
            feature = feature.view(feature.size(0), -1)
            features_list.append(feature.numpy())
    return np.vstack(features_list)

features_train = extract_features_batch(feature_extractor, X_train)
features_test = extract_features_batch(feature_extractor, X_test)

clf = BalancedRandomForestClassifier(
    replacement=True,
    sampling_strategy='not minority',
    n_estimators=300,
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(features_train, features_test)

# Step 7: Evaluate the Classifier on Test Data
y_pred = clf.predict(features_test)
print('Classifier performance on test data:')
print(metrics.classification_report(y_test, y_pred, zero_division=0, target_names=le.classes_))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Data')
plt.show()

"""
# Step 8: Save the Trained Classifier and Feature Extractor
joblib.dump(clf, 'model\\random_forest_classifier.joblib')
torch.save(feature_extractor.state_dict(), 'model\\feature_extractor.pth')
joblib.dump(le, 'model\\label_encoder.joblib')
"""
