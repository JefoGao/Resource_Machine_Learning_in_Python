# 3 Clustering
## 3.1 Kmeans - case 1
```py
# define a fake dataset
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# build a kmeans model with 2 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fix(X)

# get predicted cluster labels for each data sample
print('labels:',kmeans.labels_) # labels: [1 1 1 0 0 0]

# make cluster prediction for given data
print('prediction:',kmeans.predict([[0, 0], [12, 3]])) # prediction: [1 0]

# get cluster center for each cluster
print('kmeans.cluster_centers_:',kmeans.cluster_centers_)
# kmeans.cluster_centers_: [[10.  2.] [ 1.  2.]]
```
## 3.1 Kmeans - case 2
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# #############################################################################
# Generate sample data
np.random.seed(0)
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7) # make_blobs() function can be used to generate blobs of points with a Gaussian distribution. 

# #############################################################################
# Compute clustering with Means

kmeans = KMeans(n_clusters=3, n_init=15, random_state=0) # n_init: number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

kmeans.fit(X)
kmeans_cluster_centers = kmeans.cluster_centers_
kmeans_labels = kmeans.labels_
print(kmeans_cluster_centers)

# [[ 1.07621886 -1.06625689]
# [-1.07094261 -1.00512907]
# [ 0.96903436  1.02005354]]
```
we can find that the cluster centres found by KMeans are very close to our original generated data samples [[1, 1], [-1, -1], [1, -1]]
```py
# Plotting
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = kmeans_labels == k
    cluster_center = kmeans_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
```
![image](https://github.com/JefoGao/Resource_Machine_Learning_in_Python/assets/19381768/a99df72a-ac1f-4ae6-bc70-7c0f88e3596d)
```py
# check how many data samples in each cluster
unique_labels, unique_counts = np.unique(kmeans_labels, return_counts=True)
dict(zip(unique_labels, unique_counts))

# {0: 960, 1: 1016, 2: 1024}
```
