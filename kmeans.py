from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#so we have 2 clusters, To show the clusters
print(kmeans.labels_)
#o/p [0 0 0 1 1 1]
#so there are cluster 0 and cluster 1
#[1, 2],[1, 4], [1, 0] these points are in cluster0
#[4, 2], [4, 4], [4, 0] these points are in cluster1
print(kmeans.predict([[0, 0]]))#cluster0
print(kmeans.predict([[4, 4]]))#cluster1
#It will give you centers of clusters
print(kmeans.cluster_centers_)