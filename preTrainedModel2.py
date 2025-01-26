import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from ImuDataImport import butter_lowpass_filter

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
repo = 'OxWearables/ssl-wearables'
harnet10 = torch.hub.load(repo, 'harnet10', class_num=6, pretrained=True)

for param in harnet10.feature_extractor.parameters():
    param.requires_grad = True

class ActivityDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def extract_windows(data, winsize='10s'):
    X_windows, Y_labels = [], []
    for t, w in data.resample(winsize, origin='start'):
        if w.isna().any().any():
            continue
        if len(w) == 642:
            w = w.iloc[:641]
        elif len(w) != 641:
            continue
        x = w[['x', 'y', 'z']].to_numpy().T
        y = w['annotation'].mode(dropna=False).item()
        X_windows.append(x)
        Y_labels.append(y)
    X_windows = np.stack(X_windows)
    Y_labels = np.array(Y_labels)
    return X_windows, Y_labels

data = pd.read_csv('csv/myDataWithAnnotations.csv', index_col='time', parse_dates=['time'])
#butter lowpass filter
cutoff_freq = 3
for col in ['x', 'y', 'z']:
    data[col] = butter_lowpass_filter(data[col], cutoff=cutoff_freq, fs=64)

X_data, Y_labels = extract_windows(data)

# Encode Labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_encoded = le.fit_transform(Y_labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded)


batch_size = 128
train_dataset = ActivityDataset(X_train, y_train)
test_dataset = ActivityDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(np.unique(Y_encoded))
harnet10.classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
harnet10 = harnet10.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(harnet10.parameters(), lr=1e-4)

def train_model(model, train_loader, optimizer, criterion, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

train_model(harnet10, train_loader, optimizer, criterion)

# Step 4: Evaluate the Model
def evaluate_model(model, test_loader, le):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    cm = metrics.confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Data')
    plt.show()

evaluate_model(harnet10, test_loader, le)

