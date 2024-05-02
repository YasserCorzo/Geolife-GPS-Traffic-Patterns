import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from filtering import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

############################################# PCA ##################################################
dataset_path = 'code/test_full.csv'
df_filtered_china = filter_china_locations(dataset_path)
df_filtered_china = df_filtered_china.drop(['User', 'Total Time'], axis=1)
features = df_filtered_china.columns
print(features)

std_scaler = StandardScaler()
# normalize data points so that features have zero mean and unit variance
scaled_data = std_scaler.fit_transform(df_filtered_china)
scaled_df = pd.DataFrame(scaled_data, columns=features)
print(scaled_df.head())

# find the optimal number of PCs needed by measuring the explained variance ratio for each pc_num
# pc_nums that have an explained variance ratio of at least 95% are optimal
n_samples = len(scaled_df)
n_features = scaled_df.shape[1]
pc_nums = np.arange(min(n_samples, n_features) + 1)

var_ratio = []
for num in pc_nums:
  pca = PCA(n_components=num)
  pca.fit(scaled_df)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

# Analyzing the Change in Explained Variance Ratio
plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(pc_nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')
plt.show()

# retrieve optimal pc_num value
optimal_pc_num = 4 # to be filled after analyzing graph

pca = PCA(0.8)
pca.fit(scaled_df)
res = pca.transform(scaled_df)

# create dataframe for reduced dataset
pca_df = pd.DataFrame(res, columns=pca.get_feature_names_out(features))
print(pca_df)

# download reduced dataset to csv file
pca_df.to_csv('pca_dataset.csv')
