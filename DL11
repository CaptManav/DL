from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()

X = iris.data[:, :2]   # Take first 2 features

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

# Train model
kmeans.fit(X)

# Cluster labels
labels = kmeans.labels_

# Plot clusters
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
