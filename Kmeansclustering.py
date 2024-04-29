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
# mat = io.loadmat('SPM.mat')
# data = np.transpose(mat['subMatrix'])

mat1 = io.loadmat('Data.mat')
data1 = np.transpose(mat1['LD'])

# 将两个矩阵在第三维度上叠加，创建一个三维张量
# X = np.stack((data, data1), axis=2)

import tensorly as tl
from tensorly.decomposition import tucker
# 进行Tucker分解
# core, factors = tucker(X, rank=[27, 30, 2])
# with open("Y5.csv",encoding = 'utf-8') as f:
#     sc = np.loadtxt(f, str, delimiter = ",")
# sc = sc.T
#
# with open("V.csv",encoding = 'utf-8') as f:
#     cell = np.loadtxt(f, str, delimiter = ",")
# cell = cell.T
# print(cell.shape)
label = np.loadtxt("label.csv", delimiter=",", dtype=float, encoding='utf-8-sig')
y = label.astype(int)
clusters = y.max()-y.min()+1

model = NMF(n_components=20, init='random', random_state=42)
W = model.fit_transform(data1)
H = model.components_
from  sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ag = KMeans(n_clusters=clusters)

# cluster_labels = ag.fit_predict(factors[0])
cluster_labels = ag.fit_predict(W)
ari = adjusted_rand_score(label, cluster_labels)
nmi = normalized_mutual_info_score(label, cluster_labels)
acc = acc(label, cluster_labels)
np.savetxt('ZL_label.csv', cluster_labels, delimiter=',', fmt='%d')
print("after ARI = ", ari)
print("after NMI = ", nmi)
print("after ACC = ", acc)

