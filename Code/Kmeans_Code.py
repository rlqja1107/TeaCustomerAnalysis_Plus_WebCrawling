import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.cluster import KMeans

#Call from json
import json
with open('C:\\Users\\rlqja\\OneDrive\\바탕 화면\\Rpython\\data.json','r') as f:
	data=json.load(f)

data2=pd.DataFrame(data)
del data2['phon_Num']
del data2['Time']
Name=list(data2)
data2_f=pd.get_dummies(data2,prefix=Name)
ks=range(1,10)
iner_list=[]
for k in ks:
	model=KMeans(n_clusters=k)
	model.fit(data2_f)
	print("군집의 갯수가",k,'개 일때 inertia: ',model.inertia_)
	iner_list.append(model.inertia_)

plt.plot(ks,iner_list)
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.title('Inertia For Cluster')
plt.xticks(ks)
plt.show()
