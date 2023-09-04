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

```
