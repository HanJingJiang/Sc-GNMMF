import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans  # kmeans
from sklearn.cluster import SpectralClustering  # SC
from sklearn.cluster import AgglomerativeClustering  # 层次聚类
from sklearn.metrics import balanced_accuracy_score  # 平均精度

import warnings

warnings.filterwarnings("ignore")
# Camp in 8 method


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)

colors = ['#1f78b4', '#33a02c', '#00aedb', '#7b2b9b', '#ff7f0e', '#253494', '#cb181d', '#006837','#4daf4a','#8c564b','#bcbd22','#17becf','#e377c2','#9467bd','#fdae61']
# 函数用于添加行标题
#3*6
fig = plt.figure(figsize=(17, 18))
# gs = fig.add_gridspec(6, 6)
gs = fig.add_gridspec(5, 6)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
ax6 = fig.add_subplot(gs[0, 5])
ax7 = fig.add_subplot(gs[1, 0])
ax8 = fig.add_subplot(gs[1, 1])
ax9 = fig.add_subplot(gs[1, 2])
ax10 = fig.add_subplot(gs[1, 3])
ax11 = fig.add_subplot(gs[1, 4])
ax12 = fig.add_subplot(gs[1, 5])
ax13 = fig.add_subplot(gs[2, 0])
ax14 = fig.add_subplot(gs[2, 1])
ax15 = fig.add_subplot(gs[2, 2])
ax16 = fig.add_subplot(gs[2, 3])
ax17 = fig.add_subplot(gs[2, 4])
ax18 = fig.add_subplot(gs[2, 5])
ax19 = fig.add_subplot(gs[3, 0])
ax20 = fig.add_subplot(gs[3, 1])
ax21 = fig.add_subplot(gs[3, 2])
ax22 = fig.add_subplot(gs[3, 3])
ax23 = fig.add_subplot(gs[3, 4])
ax24 = fig.add_subplot(gs[3, 5])
ax25 = fig.add_subplot(gs[4, 0])
ax26 = fig.add_subplot(gs[4, 1])
ax27 = fig.add_subplot(gs[4, 2])
ax28 = fig.add_subplot(gs[4, 3])
# ax29 = fig.add_subplot(gs[4, 4])
# ax30 = fig.add_subplot(gs[4, 5])
# ax31 = fig.add_subplot(gs[5, 0])
# ax32 = fig.add_subplot(gs[5, 1])
# ax33 = fig.add_subplot(gs[5, 2])
# ax34 = fig.add_subplot(gs[5, 3])
# ax35 = fig.add_subplot(gs[5, 4])
# ax36 = fig.add_subplot(gs[5, 5])
#ScSPMT
dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/yan/yan.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
# x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/yan/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters1 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df1 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df1['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s1 = ax1.scatter(clustered_data_df1.loc[clustered_data_df1['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df1.loc[clustered_data_df1['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/yan/VK50.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T

labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/yan/yan_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters2 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df2 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df2['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s2 = ax2.scatter(clustered_data_df2.loc[clustered_data_df2['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df2.loc[clustered_data_df2['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))


dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/goolam/goolam.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
# x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/goolam/label.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters3 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df3 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df3['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s3 = ax3.scatter(clustered_data_df3.loc[clustered_data_df3['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df3.loc[clustered_data_df3['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/goolam/VK80.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/goolam/goolam_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters4 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df4 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df4['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s4 = ax4.scatter(clustered_data_df4.loc[clustered_data_df4['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df4.loc[clustered_data_df4['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/ENCODE/encode.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
# x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/ENCODE/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters5 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df5 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df5['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s5 = ax5.scatter(clustered_data_df5.loc[clustered_data_df5['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df5.loc[clustered_data_df5['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))
dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/ENCODE/VK80.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/ENCODE/encode_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters6 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df6 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df6['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s6 = ax6.scatter(clustered_data_df6.loc[clustered_data_df6['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df6.loc[clustered_data_df6['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))



dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/li/li.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
# x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/li/label.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters7 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df7 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df7['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s7 = ax7.scatter(clustered_data_df7.loc[clustered_data_df7['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df7.loc[clustered_data_df7['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/li/VK50.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/li/li_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters8 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df8 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df8['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s8 = ax8.scatter(clustered_data_df8.loc[clustered_data_df8['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df8.loc[clustered_data_df8['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))


dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-time/time.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
# x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-time/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters9 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df9 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df9['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s9 = ax9.scatter(clustered_data_df9.loc[clustered_data_df9['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df9.loc[clustered_data_df9['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-time/VK90.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-time/time_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters10 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df10 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df10['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s10 = ax10.scatter(clustered_data_df10.loc[clustered_data_df10['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df10.loc[clustered_data_df10['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-type/type.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-type/label.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters11 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df11 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df11['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s11 = ax11.scatter(clustered_data_df11.loc[clustered_data_df11['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df11.loc[clustered_data_df11['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))
dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-type/VK50.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/chu-type/type_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters12 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df12 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df12['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s12 = ax12.scatter(clustered_data_df12.loc[clustered_data_df12['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df12.loc[clustered_data_df12['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/kolod/kolo.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
# x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/kolod/label.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters13 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df13 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df13['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s13 = ax13.scatter(clustered_data_df13.loc[clustered_data_df13['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df13.loc[clustered_data_df13['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/kolod/VK20.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/kolod/kolo_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters14 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df14 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df14['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s14 = ax14.scatter(clustered_data_df14.loc[clustered_data_df14['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df14.loc[clustered_data_df14['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))


dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/zeisel/zeisel.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/zeisel/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters15 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df15 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df15['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s15 = ax15.scatter(clustered_data_df15.loc[clustered_data_df15['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df15.loc[clustered_data_df15['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/zeisel/VK20.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/zeisel/zeisel_pred.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters16 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df16 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df16['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s16 = ax16.scatter(clustered_data_df16.loc[clustered_data_df16['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df16.loc[clustered_data_df16['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human1/human1.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human1/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters17 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df17 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df17['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s17 = ax17.scatter(clustered_data_df17.loc[clustered_data_df17['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df17.loc[clustered_data_df17['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))
dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human1/VK150.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human1/scGNNMF.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters18 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df18 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df18['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s18 = ax18.scatter(clustered_data_df18.loc[clustered_data_df18['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df18.loc[clustered_data_df18['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human2/human2.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human2/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters19 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df19 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df19['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s19 = ax19.scatter(clustered_data_df19.loc[clustered_data_df19['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df19.loc[clustered_data_df19['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human2/VK450.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human2/scGNNMF.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters20 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df20 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df20['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s20 = ax20.scatter(clustered_data_df20.loc[clustered_data_df20['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df20.loc[clustered_data_df20['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))


dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human3/human3.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human3/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters21 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df21 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df21['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s21 = ax21.scatter(clustered_data_df21.loc[clustered_data_df21['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df21.loc[clustered_data_df21['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))
dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human3/VK100.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human3/scGNNMF.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters22 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df22 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df22['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s22 = ax22.scatter(clustered_data_df22.loc[clustered_data_df22['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df22.loc[clustered_data_df22['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human4/human4.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human4/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters23 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df23 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df23['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s23 = ax23.scatter(clustered_data_df23.loc[clustered_data_df23['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df23.loc[clustered_data_df23['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))
dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human4/VK200.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/human4/scGNNMF.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters24 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df24 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df24['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s24 = ax24.scatter(clustered_data_df24.loc[clustered_data_df24['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df24.loc[clustered_data_df24['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse1/mouse1.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse1/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters25 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df25 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df25['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s25 = ax25.scatter(clustered_data_df25.loc[clustered_data_df25['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df25.loc[clustered_data_df25['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse1/VK300.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse1/scGNNMF.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters26 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df26 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df26['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s26 = ax26.scatter(clustered_data_df26.loc[clustered_data_df26['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df26.loc[clustered_data_df26['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))


dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse2/mouse2.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse2/label.txt'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters27 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df27 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df27['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s27 = ax27.scatter(clustered_data_df27.loc[clustered_data_df27['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df27.loc[clustered_data_df27['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))
dataname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse2/VK550.csv'
with open(dataname, encoding='utf-8') as f:
    x = np.loadtxt(f, delimiter=",")
x = x.T
labelname = r'/Users/hanjing/Desktop/对比结果/Sc-GNNMF/mouse2/scGNNMF.csv'
with open(labelname, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")
y = y.astype(int)
clusters28 = y.max() - y.min() + 1
reduced_data = tsne.fit_transform(x)
clustered_data_df28 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
clustered_data_df28['Cluster'] = y
for label in set(y):
    if label == -1:
        cidx = -1
    else:
        cidx = label % len(colors)
    s28 = ax28.scatter(clustered_data_df28.loc[clustered_data_df28['Cluster'] == label, 'Dimension 1'],
                     clustered_data_df28.loc[clustered_data_df28['Cluster'] == label, 'Dimension 2'],
                     c=colors[cidx], marker='.', label='Cluster ' + str(label))

# # seurat
# labelname = r'/Users/hanjing/Desktop/对比结果/seurat/klein.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters29 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df29 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df29['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s29 = ax29.scatter(clustered_data_df29.loc[clustered_data_df29['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df29.loc[clustered_data_df29['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))
# # CORR
# labelname = r'/Users/hanjing/Desktop/对比结果/CORR/klein.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters30 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df30 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df30['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s30 = ax30.scatter(clustered_data_df30.loc[clustered_data_df30['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df30.loc[clustered_data_df30['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))
#
# #ScSPMT
# dataname = r'/Users/hanjing/Desktop/对比的方法/data/10x-5cl.csv'
# with open(dataname, encoding='utf-8') as f:
#     x = np.loadtxt(f, delimiter=",")
# # x = x.T
# labelname = r'/Users/hanjing/Desktop/ScCheb下游分析/10x-5cl/ZL_label.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters31 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df31 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df31['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s31 = ax31.scatter(clustered_data_df31.loc[clustered_data_df31['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df31.loc[clustered_data_df31['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))
#
# labelname = r'/Users/hanjing/Desktop/对比结果/POCR/10x-5cl.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters32 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df32 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df32['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s32 = ax32.scatter(clustered_data_df32.loc[clustered_data_df32['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df32.loc[clustered_data_df32['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))
#
#
# # ZIFA
# labelname = r'/Users/hanjing/Desktop/对比结果/ZIFA/10x-5cl.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters33 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df33 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df33['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s33 = ax33.scatter(clustered_data_df33.loc[clustered_data_df33['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df33.loc[clustered_data_df33['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))
# # scCAN
# labelname = r'/Users/hanjing/Desktop/对比结果/scCAN1/10x-5cl.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters34 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df34 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df34['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s34 = ax34.scatter(clustered_data_df34.loc[clustered_data_df34['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df34.loc[clustered_data_df34['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))
#
# # seurat
# labelname = r'/Users/hanjing/Desktop/对比结果/seurat/10x-5cl.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters35 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df35 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df35['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s35 = ax35.scatter(clustered_data_df35.loc[clustered_data_df35['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df35.loc[clustered_data_df35['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))
# # CORR
# labelname = r'/Users/hanjing/Desktop/对比结果/CORR/10x-5cl.csv'
# with open(labelname, encoding='utf-8') as f:
#     y = np.loadtxt(f, delimiter=",")
# y = y.astype(int)
# clusters36 = y.max() - y.min() + 1
# reduced_data = tsne.fit_transform(x)
# clustered_data_df36 = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
# clustered_data_df36['Cluster'] = y
# for label in set(y):
#     if label == -1:
#         cidx = -1
#     else:
#         cidx = label % len(colors)
#     s36 = ax36.scatter(clustered_data_df36.loc[clustered_data_df36['Cluster'] == label, 'Dimension 1'],
#                      clustered_data_df36.loc[clustered_data_df36['Cluster'] == label, 'Dimension 2'],
#                      c=colors[cidx], marker='.', label='Cluster ' + str(label))


ax1.set_title('Yan')
ax2.set_title('scGNMMF')
ax3.set_title('Goolam')
ax4.set_title('scGNMMF')
ax5.set_title('ENCODE')
ax6.set_title('scGNMMF')
ax7.set_title('Li')
ax8.set_title('scGNMMF')
ax9.set_title('Chu-time')
ax10.set_title('scGNMMF')
ax11.set_title('Chu-type')
ax12.set_title('scGNMMF')
ax13.set_title('Kolod')
ax14.set_title('scGNMMF')
ax15.set_title('Zeisel')
ax16.set_title('scGNMMF')
ax17.set_title('Human1')
ax18.set_title('scGNMMF')
ax19.set_title('Human2')
ax20.set_title('scGNMMF')
ax21.set_title('Human3')
ax22.set_title('scGNMMF')
ax23.set_title('Human4')
ax24.set_title('scGNMMF')
ax25.set_title('Mouse1')
ax26.set_title('scGNMMF')
ax27.set_title('Mouse2')
ax28.set_title('scGNMMF')
# ax29.set_title('Seurat')
# ax30.set_title('CORR')
# ax31.set_title('ScSPMT')
# ax32.set_title('POCR')
# ax33.set_title('ZIFA')
# ax34.set_title('scCAN')
# ax35.set_title('Seurat')
# ax36.set_title('CORR')
# 调整子图之间的间隔
plt.tight_layout(w_pad=1, h_pad=1)

plt.savefig('2D-compare-ScSPMT1.pdf', dpi=400, facecolor='white')
plt.savefig('2D-compare-ScSPMT1.png', dpi=400, facecolor='white')
plt.show()