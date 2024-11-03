import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


n_samples = 10000  
n_clusters = 3     
iterations = 88    

R_values = []

for i in range(iterations):

    data = np.random.normal(loc=0, scale=1, size=(n_samples, 2)) 

    kmeans = KMeans(n_clusters=n_clusters, random_state=i)
    kmeans.fit(data)

    centroids = kmeans.cluster_centers_
    norms = np.linalg.norm(centroids, axis=1)  

    R_i = np.mean(norms)
    R_values.append(R_i)


cumulative_averages = np.cumsum(R_values) / np.arange(1, iterations + 1)

plt.plot(range(1, iterations + 1), cumulative_averages, marker='o', color='b', markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Cumulative Average of R")
plt.title("Convergence of Cumulative Averages of R Values")
plt.grid(True)
plt.show()

