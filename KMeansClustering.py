from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('~/Downloads/yc-p023/YC-P0230001_6D.tsv', delimiter='\t', skiprows=41)
data = data[['r_hand_rigid_body X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']]

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Number of clusters - this should be chosen based on domain knowledge or using techniques like the Elbow method
n_clusters = 1

# K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Spectral Clustering
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral.fit_predict(data_scaled)

# Evaluate the clustering performance using silhouette scores
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
spectral_silhouette = silhouette_score(data_scaled, spectral_labels)

print(f"K-Means Silhouette Score: {kmeans_silhouette}")
print(f"Spectral Clustering Silhouette Score: {spectral_silhouette}")

# Identify potential anomalies (data points that are distant from cluster centroids in K-Means)
distances = kmeans.transform(data_scaled)  # distances to cluster centers
closest = np.min(distances, axis=1)  # closest distance to the centers
outliers = closest > np.percentile(closest, 95)  # thresholding the distance

# Plotting anomalies detected by K-Means
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=outliers, cmap='coolwarm', label='Normal vs Anomaly')
plt.title('Anomalies detected by K-Means (Red Points)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()