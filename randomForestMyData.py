# Code is partially based on the following source: https://github.com/OxWearables/capture24/blob/master/tutorial.ipynb

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

import features
from ImuDataImport import butter_lowpass_filter

np.random.seed(42)


data = pd.read_csv('csv\\myDataWithAnnotations.csv', index_col='time', parse_dates=['time'])

print("\nUnique annotations")
print(pd.Series(data['annotation'].unique()))

#butter lowpass filter
cutoff_freq = 3
for col in ['x', 'y', 'z']:
    data[col] = butter_lowpass_filter(data[col], cutoff=cutoff_freq, fs=64)

def extract_windows(data, winsize='10s'):
    X, Y = [], []
    for t, w in data.resample(winsize, origin='start'):

        # Check window has no NaNs and is of correct length
        # 10s @ 64Hz = 640 ticks
        if w.isna().any().any():
            print(f"Skipping window at {t} due to NaNs")
            continue

        if len(w) == 642:
            w = w.iloc[:641]
        elif len(w) != 641:
            print(f"Skipping window at {t} with {len(w)} samples")
            continue

        x = w[['x', 'y', 'z']].to_numpy()
        y = w['annotation'].mode(dropna=False).item()

        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y

# %%

print("Extracting windows...")
X, Y = extract_windows(data)

# %% [markdown]
'''
## Visualization
Let's plot some instances of each activity

# %%

# Plot activities
print("Plotting activities...")
NPLOTS = 5
unqY = np.unique(Y)
fig, axs = plt.subplots(len(unqY), NPLOTS, sharex=True, sharey=True, figsize=(10,10))
for y, row in zip(unqY, axs):
    idxs = np.random.choice(np.where(Y==y)[0], size=NPLOTS)
    row[0].set_ylabel(y)
    for x, ax in zip(X[idxs], row):
        ax.plot(x)
        ax.set_ylim(-5,5)
fig.tight_layout()
fig.show()

# %% [markdown]


Something to note from the plot above is the heterogeneity of the signals even
within the same activities. This is typical of free-living data, as opposed to
clean lab data where subjects perform a scripted set of activities under
supervision.

Now let's perform a PCA plot


# %%

# PCA plot
print("Plotting first two PCA components...")
scaler = preprocessing.StandardScaler()  # PCA requires normalized data
X_scaled = scaler.fit_transform(X.reshape(X.shape[0],-1))
pca = decomposition.PCA(n_components=2)  # two components
X_pca = pca.fit_transform(X_scaled)

NPOINTS = 200
unqY = np.unique(Y)
fig, ax = plt.subplots()
for y in unqY:
    idxs = np.random.choice(np.where(Y==y)[0], size=NPOINTS)
    x = X_pca[idxs]
    ax.scatter(x[:,0], x[:,1], label=y, s=10)
fig.legend()
fig.show()

# %% [markdown]

## Activity recognition

We will use a random forest to build the activity recognition model.
Rather than using the raw signal as the input which is very inefficient,
we use common statistics and features of the signal such as the quantiles,
correlations, dominant frequencies, number of peaks, etc. See `features.py`.


Note: this may take a while

'''

# %%

X_feats = pd.DataFrame([features.extract_features(x) for x in X])
sample_features = features.extract_features(X[0])
print(f"Number of features extracted: {len(sample_features)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_feats, Y, test_size=0.2, random_state=42, stratify=Y
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Define the resampling strategy
oversampler = SMOTE(random_state=42)
undersampler = RandomUnderSampler(random_state=42)
combiner = SMOTEENN(random_state=42)

X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
print(f"Original class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Resampled class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

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
clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred, zero_division=0))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Data')
plt.show()

"""
print('Training set class distribution:')
print(pd.Series(y_train).value_counts())

print('Testing set class distribution:')
print(pd.Series(y_test).value_counts())

y_train_pred = clf.predict(X_train)
print('\nClassifier performance on training data')
print(metrics.classification_report(y_train, y_train_pred, zero_division=0))



importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(f"Total features: {len(feature_importance_df)}")

f1_scores = []
for N in range(5, len(feature_importance_df) + 1, 5):  # Increment by 5
    top_features = feature_importance_df['Feature'].head(N).tolist()
    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]

    clf.fit(X_train_reduced, y_train)
    y_pred_reduced = clf.predict(X_test_reduced)

    f1 = metrics.f1_score(y_test, y_pred_reduced, average='weighted', zero_division=0)
    f1_scores.append((N, f1))


plt.plot([score[0] for score in f1_scores], [score[1] for score in f1_scores])
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Number of Features")
plt.show()


from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}


clf = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)


grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all available processors
    verbose=1
)


grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Hyperparameters: {best_params}")
print(f"Best F1-Weighted Score: {best_score}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred, zero_division=0))
"""
