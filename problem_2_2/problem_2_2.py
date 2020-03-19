import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from time import time
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE 
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

#take subset of data so it doesn't take forever 
X_shuffle, y_shuffle = shuffle(X, y, random_state=0)
X_trim = X_shuffle[0:10000]
y_trim = y_shuffle[0:10000]
#print(y_trim)
#print(X_trim.shape)

colors = ['b', 'g', 'k', 'c', 'm', 'y', 'w']
non_cost = []
pca_cost = []
for clusters in range(8,9):
	k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=42, n_init=10)
	t0 = time()
	k_means_x = k_means.fit(X_trim)
	non_cost.append(k_means_x.inertia_)
	print("Fit compute time " + str(time()-t0) + " with " + str(clusters) +  " clusters")

	cluster_labels = range(0, clusters)
	
	plt.figure()
	ax = plt.subplot(211)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)

	tsne = TSNE(n_components=2, random_state=42)
	assigned_cluster = k_means_x.predict(X_trim)
	combined = np.append(X_trim, k_means_x.cluster_centers_, axis=0)
	#print(combined.shape)
	x_transformed = tsne.fit_transform(combined)
	x_min, x_max = np.min(x_transformed, 0), np.max(x_transformed, 0)
	x_transformed = (x_transformed - x_min) / (x_max - x_min)
	#print(x_transformed.shape)
	for i in range(X_trim.shape[0]):
		size = 9
		color = colors[assigned_cluster[i] % 7]
		plt.text(x_transformed[i,0], x_transformed[i,1], str(y_trim[i]),
        		 color=color,
                 fontdict={'weight': 'bold', 'size': size})
	for i in range(X_trim.shape[0], X_trim.shape[0]+k_means_x.cluster_centers_.shape[0]):
		size = 20
		color = colors[(i - X_trim.shape[0]) % 7]
		plt.text(x_transformed[i,0], x_transformed[i,1], str(i - X_trim.shape[0]),
        		 color=color,
                 fontdict={'weight': 'bold', 'size': size})

	#print("PCA reduced")
	t0 = time()
	reduced_data = PCA(n_components=5).fit_transform(X_trim)
	kmeans = KMeans(init='k-means++', n_clusters=clusters, random_state=42, n_init=10)
	kmeans.fit(reduced_data)
	pca_cost.append(kmeans.inertia_)
	print("Fit with PCA compute time " + str(time()-t0) + " with " + str(clusters) + " clusters")
	
	ax = plt.subplot(212)
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)

	tsne = TSNE(n_components=2, random_state=42)
	assigned_cluster = kmeans.predict(reduced_data)
	combined = np.append(reduced_data, kmeans.cluster_centers_, axis=0)
	#print(combined.shape)
	x_transformed = tsne.fit_transform(combined)
	x_min, x_max = np.min(x_transformed, 0), np.max(x_transformed, 0)
	x_transformed = (x_transformed - x_min) / (x_max - x_min)
	#print(x_transformed.shape)
	for i in range(reduced_data.shape[0]):
		size = 9
		color = colors[assigned_cluster[i] % 7]
		plt.text(x_transformed[i,0], x_transformed[i,1], str(y_trim[i]),
        		 color=color,
                 fontdict={'weight': 'bold', 'size': size})
	for i in range(reduced_data.shape[0], reduced_data.shape[0]+k_means_x.cluster_centers_.shape[0]):
		size = 20
		color = colors[(i - reduced_data.shape[0]) % 7]
		plt.text(x_transformed[i,0], x_transformed[i,1], str(i - reduced_data.shape[0]),
        		 color=color,
                fontdict={'weight': 'bold', 'size': size})


    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
	silhouette_avg = silhouette_score(reduced_data, assigned_cluster)
	print("For n_clusters =", clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
	sample_silhouette_values = silhouette_samples(reduced_data, assigned_cluster)

	#ax1 = plt.subplot(111)
	#y_lower = 10
	#for i in range(clusters):
	#	# Aggregate the silhouette scores for samples belonging to
	#	# cluster i, and sort them
	#	ith_cluster_silhouette_values = \
	#	sample_silhouette_values[assigned_cluster == i]

	#	ith_cluster_silhouette_values.sort()

	#	size_cluster_i = ith_cluster_silhouette_values.shape[0]
	#	y_upper = y_lower + size_cluster_i

	#	color = cm.nipy_spectral(float(i) / clusters)
	#	ax1.fill_betweenx(np.arange(y_lower, y_upper),
	#					  0, ith_cluster_silhouette_values,
	#					  facecolor=color, edgecolor=color, alpha=0.7)

    #   # Label the silhouette plots with their cluster numbers at the middle
	#	ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
	#	y_lower = y_upper + 10  # 10 for the 0 samples	

	#ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.show()

#plt.plot(range(2, 31), non_cost, color ='g', linewidth ='3') 
#plt.plot(range(2, 31), pca_cost, color ='g', linewidth ='3') 
#plt.xlabel("Value of K") 
#plt.ylabel("Sqaured Error (Cost)") 
#plt.show() # clear the plot 
print("done")