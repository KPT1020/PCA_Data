import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

 #columns need to isolate the first 0,0 data
sensor=['MQ2','MQ4','MQ5','MQ6','MQ7','MQ8','MQ9']
data=pd.read_csv('mixed_ppm.csv')
data=data.loc[0:39]
data=data.drop(['ppm','gas','MQ9'],axis=1)
print(data)

scaled_data = preprocessing.scale(data) #dont transpose_data
# print(data)

pca=PCA()
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
#mutiply the ratio by 100 and round to 1 decimal
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
#there should be 6 in total
pca_plot=pd.DataFrame(pca_data,columns=labels)
print(pca_plot)


for i in range(0,39):
    if i <=7:
        plt.scatter(pca_plot.loc[i,'PC1'], pca_plot.loc[i,'PC2'],c='red')
    if i>7 and i<=15:
        plt.scatter(pca_plot.loc[i,'PC1'], pca_plot.loc[i,'PC2'],c='blue')
    if i>15 and i<=23:
        plt.scatter(pca_plot.loc[i,'PC1'], pca_plot.loc[i,'PC2'],c='green')
    if i>23 and i<=31:
        plt.scatter(pca_plot.loc[i,'PC1'], pca_plot.loc[i,'PC2'],c='yellow')
    if i>31 and i<=39:
        plt.scatter(pca_plot.loc[i,'PC1'], pca_plot.loc[i,'PC2'],c='coral')

plt.title('show_me_sth')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))#give the cprrelation of first data
plt.ylabel('PC2 - {0}%'.format(per_var[1]))#give the correlation of the second

for sample in pca_plot.index :
    plt.annotate(sample, (pca_plot.PC1.iloc[sample], pca_plot.PC2.iloc[sample]))
 
plt.show()

# dominant_sensor= pd.Series(pca.components_[0], index=[ppm])
# ## now sort the loading scores based on their magnitude
# sorted_loading_scores = dominant_sensor.abs().sort_values(ascending=False)
 
# # get the names of the top 10 genes
# # top_10_genes = sorted_loading_scores.index.values
 
# ## print the gene names and their scores (and +/- sign)
# print('The dominant factor is {}'.format(sorted_loading_scores.index[0]))
