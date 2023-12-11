import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Loads a dataset containing images of handwritten numbers from 0 to 9.
digits = load_digits()

# Scales feature values.
# This is an important step in preprocessing data before using it in machine learning algorithms.
# Scaling is usually done to ensure that all features have the same scale,
# which helps models learn better.
data = scale(digits.data)

# print(data)
# define target data
y = digits.target


# number of centroids
k = 10

# k = len(np.unique(y))


# In this code, the line samples, features = data.shape assigns the samples and features variables the values
# corresponding to the number of rows and columns in the data array.
# In this context, data.shape returns a tuple (number of rows, number of columns)
# for the data array - probably for plotting
samples, features = data.shape

# Trains the passed clustering algorithm (estimator) on the input data (data) using the fit method.
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
            # inertia_ - The sum of squared distances from each data point to the nearest centroid.
            
          
            # All these parameters will eventually be output to the console.
            # Homogeneity of clusters.  
            metrics.homogeneity_score(y, estimator.labels_),
            # Completeness of clusters.
            metrics.completeness_score(y, estimator.labels_),
            # Harmonic mean between uniformity and completeness.
            metrics.v_measure_score(y, estimator.labels_),
            #Adjusted Rand index. The Rand Index is a measure of the similarity between two clusterings.
            # It measures how consistent two clusterings are regarding pairs of data points.
            # The Rand index is defined in the range from 0 to 1, where 0 means complete random matching,
            # and 1 means full agreement.
            metrics.adjusted_rand_score(y, estimator.labels_),
            # Adjusted mutual information.
            metrics.adjusted_mutual_info_score(y,  estimator.labels_),
            # Silhouette coefficient for each data point.
            metrics.silhouette_score(data, estimator.labels_,
                                       metric="euclidean")))
    

# These metrics are used to evaluate how well the clustering algorithm separates
# data for clusters.




classifier = KMeans(n_clusters=k, init="random", n_init=10)
# n_clusters=k - the number of centroids, respectively the number of classes
# init="random" - method for placing centroids
# The n_init parameter in the KMeans algorithm determines how many times the algorithm will be run
# with different starting centroids.

# pass all specified parameters to the bench_k_means function
bench_k_means(classifier, "name", data)