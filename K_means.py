import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

wcss = []
from sklearn.cluster import KMeans
"""
for i in range (1,11):
    kmeans = KMeans(n_clusters= i , init = 'k-means++',max_iter=300 )
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()"""



kmeans = KMeans(n_clusters= 5 , init = 'k-means++',max_iter=300 ,random_state=0)
y_means = kmeans.fit_predict(x)

#visualising data
plt.scatter(x[y_means == 0,0],x[y_means == 0, 1], s = 100 ,c= 'red',label ='cluster 1',alpha=0.2)
plt.scatter(x[y_means == 1,0],x[y_means == 1, 1], s = 100 ,c= 'cyan',label ='cluster 2',alpha=0.2)
plt.scatter(x[y_means == 2,0],x[y_means == 2, 1], s = 100 ,c= 'green',label ='cluster 3',alpha=0.2)
plt.scatter(x[y_means == 3,0],x[y_means == 3, 1], s = 100 ,c= 'yellow',label ='cluster 4',alpha=0.2)
plt.scatter(x[y_means == 4,0],x[y_means == 4, 1], s = 100 ,c= 'magenta',label ='cluster 5',alpha=0.2)
plt.scatter(kmeans.cluster_centers_[:,0] , kmeans.cluster_centers_[:,1], s= 300 , c = 'black' , label='centroids'  )
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
