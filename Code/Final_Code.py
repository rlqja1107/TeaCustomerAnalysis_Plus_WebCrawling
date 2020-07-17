import pandas as pd
import json
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
with open('C:\\Users\\rlqja\\OneDrive\\바탕 화면\\Rpython\\data.json','r') as f:
	data2=json.load(f)
data=pd.DataFrame(data2)
data.columns=['time','sex','age','occupation','Wine','When','Caffein','Snack','Hear','efficacy','phoneNum']
del data['phoneNum']
del data['time']
del data['Hear']
data['sex']=data['sex'].apply(lambda x: 1 if x=="남" else 0)
data['Caffein']=data['Caffein'].apply(lambda x: 1 if x=="예, 선호합니다" else 2 if x=='상관없습니다.' else 0)
data['age']=data['age'].apply(lambda x:20 if x=='20대' else 10 if x=='10대' else 30 if x=='30대' else 40 if x=='40대' else 50 if x=='50대 이상' else x)
data.replace({'occupation':'대학생'},{'occupation':'0'},inplace=True)
data.replace({'occupation':'직장인'},{'occupation':'1'},inplace=True)
data.replace({'occupation':'기타'},{'occupation':'2'},inplace=True)
data.replace({'Wine':'산뜻하고 상큼함이 느껴지는 화이트와인'},{'Wine':'0'},inplace=True)
data.replace({'Wine':'무게감있고 풍미가 진한 레드와인'},{'Wine':'1'},inplace=True)
data.replace({'When':'기상 직후'},{'When':0},inplace=True)
data.replace({'When':'하루 중 여가시간'},{'When':1},inplace=True)
data.replace({'When':'식사 후'},{'When':2},inplace=True)
data.replace({'When':'취침 전'},{'When':3},inplace=True)
data.replace({'Caffein':'예, 선호합니다'},{'Caffein':0},inplace=True)
data.replace({'Caffein':'상관없습니다.'},{'Caffein':2},inplace=True)
data.replace({'Caffein':'아니요, 카페인이 꼭 필요합니다.'},{'Caffein':1},inplace=True)
data.replace({'efficacy':'노화방지, 피부미용'},{'efficacy':0},inplace=True)
data.replace({'efficacy':'소화불량 개선'},{'efficacy':1},inplace=True)
data.replace({'efficacy':'심신안정'},{'efficacy':2},inplace=True)
data.replace({'efficacy':'원기회복 및 피로회복'},{'efficacy':3},inplace=True)
data.replace({'efficacy':'집중력'},{'efficacy':4},inplace=True)
data.replace({'Snack':'쿠키'},{'Snack':0},inplace=True)
data.replace({'Snack':'마카롱'},{'Snack':1},inplace=True)
data.replace({'Snack':'케이크'},{'Snack':2},inplace=True)
data.replace({'Snack':'떡'},{'Snack':3},inplace=True)

dend=sch.dendrogram(sch.linkage(data,method='ward'))
hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(data)
y_hc=pd.DataFrame(y_hc)

result=pd.concat([data,y_hc],axis=1)
result.columns=['sex', 'age', 'occupation', 'Wine', 'When', 'Caffein', 'Snack', 'efficacy', 'Cluster']
result0=result[result['Cluster']==0]
result1=result[result['Cluster']==1]
result2=result[result['Cluster']==2]
result3=result[result['Cluster']==3]

col=['sex','age','occupation','Wine','When','Caffein','Snack','efficacy','Cluster']

temp=[]
num=0
predict=pd.DataFrame(columns=('sex','age','occupation','Wine','When','Caffein','Snack','efficacy','Cluster'))
result_clu=[result0,result1,result2,result3]

for resul in result_clu:
      temp=[]
      for index in col:
            temp.append(resul[index].value_counts().index.values[0])
      predict.loc[num]=temp
      num=num+1


