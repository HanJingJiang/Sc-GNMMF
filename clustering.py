import numpy as np
import scipy.io
from scipy import sparse
import math
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

import warnings
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import AgglomerativeClustering
import scipy.io as sio
warnings.filterwarnings('ignore')

from scipy import io#2000*3918
import h5py
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size




np.random.seed(42)


from sklearn.metrics import silhouette_score, silhouette_samples
warnings.filterwarnings('ignore')
label = np.loadtxt("label.txt", delimiter=",", dtype=float, encoding='utf-8-sig')
data = np.loadtxt("VK80.csv", delimiter=",", dtype=float, encoding='utf-8-sig')
print(data.shape)
data = data.T
from  sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
y = label.astype(int)
num_clusters = y.max()-y.min()+1
kmeans = AgglomerativeClustering(n_clusters=13)
cluster_labels = kmeans.fit_predict(data)
ari = adjusted_rand_score(label, cluster_labels)
nmi = normalized_mutual_info_score(label, cluster_labels)
acc = acc(label, cluster_labels)
print("after ARI = ", ari)
print("after NMI = ", nmi)
print("after ACC = ", acc)

