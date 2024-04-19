import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os

from filtering import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

############################################# PCA ##################################################
dataset_path = 'test.csv'
df_filtered_china = filter_china_locations(dataset_path)

std_scaler = StandardScaler()
# normalize data points so that features have zero mean and unit variance
scaled_df = std_scaler.fit_transform(df_filtered_china)

# find the optimal number of PCs needed by measuring the explained variance ratio for each pc_num
# pc_nums that have an explained variance ratio of at least 95% are optimal
pc_nums = np.arange(14)

var_ratio = []
for num in pc_nums:
  pca = PCA(n_components=num)
  pca.fit(scaled_df)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

# Analyzing the Change in Explained Variance Ratio
plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

# retrieve optimal pc_num value
optimal_pc_num = None # to be filled after analyzing graph
