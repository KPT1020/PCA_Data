import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

data=pd.read_csv('C:\\Users\\chich\\Gtihub\\PCA_Data\\Testing excelto csv.csv',dtype={"date":str,"age":np.int32})
#data= pd.DataFrame(data,index= list('123456789'))
print(data)
data_sort = data.sort_values(by='age',ascending= False)# sorting data with different columns
print(data_sort)
data_average = data.age.mean() #calculate the mean
data_average = data_average.round(1) # round off double varaible
print(data_average)

# calling several column in the data set
print(data.loc[2:5,'age']) 
#:number take rows of data #::number take all with every data
#[number:number]while comma give a range of columns and rows of the data require