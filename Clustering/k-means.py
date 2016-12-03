from sklearn.cluster import KMeans
import time

# Setting k size with average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5


kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

end = time.time()
elapsed = end - start

print "Time taken for K Means clustering: ", elapsed, "seconds."
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.index2word, idx ))
