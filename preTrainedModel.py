import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt

repo = 'OxWearables/ssl-wearables'
harnet10 = torch.hub.load(repo, 'harnet10', class_num=6, pretrained=True)

feature_extractor = nn.Sequential(*list(harnet10.children())[:-1])
feature_extractor.eval()


class CustomModel(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(CustomModel, self).__init__()
        self.feature_extractor = feature_extractor


        with torch.no_grad():
            sample_input = torch.randn(1, 3, 641)  # Adjust sequence length if necessary
            sample_output = self.feature_extractor(sample_input)
            feature_dim = sample_output.view(1, -1).shape[1]


        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

model = CustomModel(feature_extractor, 6)

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
X_data, Y_labels = extract_windows(data)

X_data = torch.tensor(X_data, dtype=torch.float32)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y_labels)
Y_encoded = torch.tensor(Y_encoded, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, Y_encoded, test_size=0.2, random_state=7, stratify=Y_encoded
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
batch_size = 128

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x_batch.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    epoch_loss /= len(X_train)
    epoch_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')


model.eval()
with torch.no_grad():
    y_pred_list = []
    y_true_list = []
    for i in range(0, len(X_test), batch_size):
        x_batch = X_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]

        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)

        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(y_batch.cpu().numpy())

    print('Classifier performance on test data:')
    print(metrics.classification_report(y_true_list, y_pred_list, zero_division=0, target_names=le.classes_))


    cm = metrics.confusion_matrix(y_true_list, y_pred_list)


    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Data')
    plt.show()

torch.save(model.state_dict(), 'model\\custom_model.pth')
joblib.dump(le, 'model\\label_encoder.joblib')
