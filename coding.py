# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

df = pd.read_csv('Mall_Customers.csv') 

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Drop CustomerID (not relevant for clustering)
df.drop(columns=['CustomerID'], inplace=True)

# Convert Gender to numerical (Male=0, Female=1)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Select features for clustering
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Scale features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use Elbow Method to determine optimal number of clusters
inertia = []
K_range = range(1, 10)  # Test clusters from 1 to 10
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Choose optimal K (Example: K=5 based on Elbow Method)
optimal_K = 5
kmeans = KMeans(n_clusters=optimal_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataset
df['Cluster'] = cluster_labels

# Evaluate clustering performance using Silhouette Score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Visualize clusters in 2D using PCA
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], 
            pca.transform(kmeans.cluster_centers_)[:, 1], 
            c='red', marker='X', label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering Visualization')
plt.legend()
plt.show()

# Save clustered data to a new CSV file
df.to_csv('newdata.csv', index=False)
print("Clustered data saved as 'newdata.csv'.")
