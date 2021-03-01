import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

MQ=['MQ2','MQ4','MQ5','MQ6','MQ7','MQ8','MQ9'] #columns
sensor=['Alcohol','CO','CH4','LPG','H2','AIR']
data=pd.read_csv('PCA_extracted_data.csv')
print(data)

data=data[MQ] #sliced column of data
data.index=[sensor]
data=data.round(2)
print(data)

scaled_data = preprocessing.scale(data.T)#transpose data
print(data.T)
#____RUN_PCA___#
pca = PCA() # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
#mutiply the ratio by 100 and round to 1 decimal
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
#there should be 6 in total
pca_plot=pd.DataFrame(pca_data, index=[MQ],columns=labels)
print(pca_plot)

plt.scatter(pca_plot.PC1, pca_plot.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))#give the cprrelation of first data
plt.ylabel('PC2 - {0}%'.format(per_var[1])) #give the correlation of the second

for sample in pca_plot.index:
    plt.annotate(sample, (pca_plot.PC1.loc[sample], pca_plot.PC2.loc[sample]))
 
plt.show()

# dominant_sensor= pd.Series(pca.components_[0], index=[MQ])
# ## now sort the loading scores based on their magnitude
# sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# # get the names of the top 10 genes
# top_10_genes = sorted_loading_scores.index.values
 
# ## print the gene names and their scores (and +/- sign)
# print(loading_scores[top_10_genes])
