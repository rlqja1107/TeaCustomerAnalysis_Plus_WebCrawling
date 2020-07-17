

import urllib.request import urllib.parse
from bs4 import BeautifulSoup import re
#Q1 
keywords=urllib.parse.quote("금리")
url='https://search.naver.com/search.naver?where=news&query='+keywords+'&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=2020.04.13&de=2020. 04.14'
req=urllib.request.urlopen(url)
data=req.read()
soup=BeautifulSoup(data,'html.parser')
anchor_set=soup.findAll( 'a' )
news_link=[]
for link in anchor_set:
      if (link[ 'href' ].startswith( 'https://news.naver.com/main/read.nhn' )):
            news_link.append(link[ 'href' ])

#Q2

keywords=urllib.parse.quote( ' 이자율 ' )
url= 'https://search.naver.com/search.naver?where=news&query='+keywords+'&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=2020.04.13&de=2020. 04.14'
data=urllib.request.urlopen(url).read()
soup=BeautifulSoup(data,'html.parser')
count_tag=soup.find( 'div' ,{ 'class' , 'title_desc all_my' })
count_text=count_tag.find( 'span' ).get_text().split()
total_num=count_text[ -1 ][ 0 : -1 ].replace( "," , "" )
new_link=set()
for val in range(int(total_num)// 10 ):
      start_val=str(val* 10 + 1 )
      url_sample = 'https://search.naver.com/search.naver?where=news&query='+keywords+'&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=& pd=3&ds=2020.04.13&de=2020.04.14&docid=&nso=so:r,p:from20200413to20200414,a:all&my news=0&cluster_rank=26&start=' +start_val+ '&refresh_start=0'
# 아래 soup_sample 과 위의 url_sample 의 줄맞춤은 pdf 변환과정에서 한칸씩 앞으로 당겨짐
      soup_sample=BeautifulSoup(urllib.request.urlopen(url_sample).read(), 'html.parser' )
      anchor_set_sample=soup_sample.findAll( 'a' ) for link in anchor_set_sample:
      if (link[ 'href' ].startswith( 'https://news.naver.com/main/read.nhn' )):
            new_link.add(link[ 'href' ])

                                                                            
#Q3
keywords = urllib.parse.quote( " 금리 " )
url = 'https://search.naver.com/search.naver?where=news&query=' + keywords + '&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=2020.04.13&de=2020.0 4.14'

req = urllib.request.urlopen(url) data = req.read()

soup = BeautifulSoup(data, 'html.parser' )
anchor_set = soup.findAll( 'a' ) news_link = []

for link in anchor_set:
      if (link[ 'href' ].startswith( 'https://news.naver.com/main/read.nhn' )):
            news_link.append(link[ 'href' ])

title_list=[]
text_list=[]

for url in news_link:
      data=urllib.request.urlopen(url).read()
      soup=BeautifulSoup(data, 'html.parser' )
      title_tag=soup.find( 'h3' ,{ 'id' : 'articleTitle' }).get_text()
      text_tag=soup.find( 'div' ,{ 'id' : 'articleBodyContents' }).get_text()
      title_list.append(title_tag)
      text_list.append(text_tag)

#Q4

for title in title_list:
      if re.search('.*금리.* 인하',title):
      print(title)





#Q5
from sklearn.linear_model
import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

missing_value=[ 'na' , 'NaN' ]

# 저장위치는 각 컴퓨터마다 다를수 있음

data=pd.read_csv( 'C:\\Users\\rlqja\\OneDrive\\ 바탕 화면\\Rpython\\boston_csv.csv' ,na_values=missing_value) data.dropna(inplace= True )

#Q6

#describe method 이용
summarize=data.describe()
# 상관관계구하기
corr=data.corr(method= 'pearson' )
heat=sns.heatmap(corr) plt.show()

#Q7

x_train=data[[ 'LSTAT' ]].loc[: 502 * 0.75 ]
x_test=data[[ 'LSTAT' ]].loc[ 502 * 0.75 :]
y_train=data[ 'MEDV' ].loc[: 502 * 0.75 ]
y_test=data[ 'MEDV' ].loc[ 502 * 0.75 :]
lm=LinearRegression() lm.fit(x_train,y_train)

Yhat_train=lm.predict(x_train)

print(mean_squared_error(y_train,Yhat_train))    #MSE

print(lm.coef_) #coefficient( 계수 )
print(lm.intercept_) #y 절편 ( 상수 )

print(lm.score(x_train,y_train)) #R square

Yhat_test=pd.DataFrame(lm.coef_[ 0 ]*x_test + lm.intercept_) # 회귀분석 추정계수를 바 탕으로 예측

print(mean_squared_error(y_test,Yhat_test))

#Q8
lm=LinearRegression()

x_train2 = data[[ 'LSTAT' , 'TAX' ]].loc[: 502 * 0.75 ]
x_test2 = data[[ 'LSTAT' , 'TAX' ]].loc[ 502 * 0.75 :]
y_train2 = data[ 'MEDV' ].loc[: 502 * 0.75 ]
y_test2 = data[ 'MEDV' ].loc[ 502 * 0.75 :]

lm.fit(x_train2,y_train2)

print(lm.coef_) # 독립변수 LSTAT 와 TAX 의 계수
print(lm.intercept_) # 상수

print(lm.score(x_train2,y_train2)) #R square
Yhat2=lm.predict(x_train2)

print(mean_squared_error(y_train2,Yhat2))
Yhat2_test =pd.DataFrame(lm.coef_[ 0 ]*x_test2[ 'LSTAT' ]+lm.coef_[ 1 ]*x_test2[ 'TAX' ]+lm.intercept_)
# 회귀분석 추정계수 값을 바탕으로 예측

print(mean_squared_error(y_test2, Yhat2_test)) #mean squared error 구하기

