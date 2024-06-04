from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('~/Downloads/yc-p023/YC-P0230001_6D.tsv', delimiter='\t', skiprows=41)
data = data[['r_hand_rigid_body X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']]

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.9, min_samples=10)  # Adjust these parameters based on your data
clusters = dbscan.fit_predict(data_scaled)

# Find outlier data points
outliers = data[clusters == -1]

# Visualization
unique, counts = np.unique(clusters, return_counts=True)
cluster_sizes = dict(zip(unique, counts))
# Plotting with annotations
plt.figure(figsize=(10, 6))
for cluster_id in unique:
    if cluster_id != -1:  # Exclude noise points
        points = data_scaled[clusters == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster_id}')
        # Annotation
        plt.text(np.mean(points[:, 0]), np.mean(points[:, 1]), f'Size: {cluster_sizes[cluster_id]}',
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
plt.legend()
plt.title('DBSCAN Clustering with Cluster Sizes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Additional diagnostics
print("Number of clusters found:", len(set(clusters)) - (1 if -1 in clusters else 0))
print("Number of anomalies detected:", len(outliers))

# Assuming 'data_scaled' from previous step
neighbors = NearestNeighbors(n_neighbors=4)  # You can start with min_samples value here
neighbors_fit = neighbors.fit(data_scaled)
distances, indices = neighbors_fit.kneighbors(data_scaled)

# Sort distance values by the distance to the nth neighbor (n = min_samples)
sorted_distances = np.sort(distances[:, 3], axis=0)
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.title('K-Distance Graph (DBSCAN)')
plt.xlabel('Points sorted by distance')
plt.ylabel('4th nearest neighbor distance')
plt.grid(True)
plt.show()