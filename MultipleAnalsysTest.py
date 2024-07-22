#重回帰分析　WEBサイトのまね
#https://qiita.com/karaage0703/items/f38d18afc1569fcc0418 参考サイト

import pandas as pd
import numpy as np    #numpy
#import sklearn
#df = pd.read_excel('BostonDataSet.xlsx', sheet_name=0, header=0)
df = pd.read_excel('winequality-red.xlsx', sheet_name=0, header=0)

matrix = df.shape #行数、列数取得
print(matrix)
print(df)

#CRIM   ZN  INDUS  CHAS   NOX      RM   AGE     DIS  RAD  TAX  PTRATIO  MEDV 
"""
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 MEDV     Median value of owner-occupied homes in $1000's
 """
#print(df['CRIM'])
""" データを列毎に取得　"""
"""MEDV_d = df.loc[:,'MEDV']  #住宅価格
#print(MEDV_d[0:])
#print(df.loc[1,'CRIM'])
CRIM_d = df.loc[:,'CRIM']   #列取得
#print(CRIM_d[0])
ZN_d = df.loc[:,'ZN'] 
INDUS_d = df.loc[:,'INDUS']
CHAS_d = df.loc[:,'CHAS']
NOX_d = df.loc[:,'NOX']
RM_d = df.loc[:,'RM']
AGE_d = df.loc[:,'AGE']
DIS_d = df.loc[:,'DIS']
RAD_d = df.loc[:,'RAD']
TAX_d = df.loc[:,'TAX']
PTRATIO_d = df.loc[:,'PTRATIO']"""

#データセット
     #y=MEDV x1,x2,x=上のどれか
x = df.loc[:,["DIS","ZN"]]
y = df.loc[:,'PTRATIO']
x1 = df.loc[:,'DIS']
x2 = df.loc[:,'ZN']

print(x.shape)
print(y.shape)

###########################################################３Dグラフ化
from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import matplotlib.pyplot as plt
import seaborn as sns


fig=plt.figure()
ax=Axes3D(fig)
fig.add_axes(ax)  #これないとダメらしい
 
ax.scatter3D(x1, x2, y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
 
#plt.show()

#####################################################################標準化・正規化
#重回帰分析で、入力変数が複数になったことで正規化の必要性
#分散を用いて標準化

#numpyによる正規化　
#yがNaNになる
x_np = x.apply(lambda x: (x - np.mean(x)) / np.std(x))
#y_np = y.apply(lambda y: (y - np.mean(y)) / np.std(y))
y_np = (y - np.mean(y)) / np.std(y)

print('$$$$$$  numpyによる標準化  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print(np.mean(y))
print(np.std(y))
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

print(x_np.head())
print(y_np.head())

#pandasによる正規化
xss_pd = (x - x.mean()) / x.std(ddof=0)
yss_pd = (y - y.mean()) / y.std(ddof=0)

print('pandasによる標準化')
print(xss_pd.head())
print(yss_pd.head())

#scikit-learnによる正規化
from sklearn import preprocessing
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression

sscaler = preprocessing.StandardScaler()
#x_fit =np.array(x).reshape(-1,1) #これで変換しないとfitに入力できない
x_fit = x
y_fit = np.array(y).reshape(-1,1) #これで変換しないとfitに入力できない

sscaler.fit(x_fit)
xss_sk = sscaler.transform(x_fit) 
sscaler.fit(y_fit)
yss_sk = sscaler.transform(y_fit)

print('scikit-learnによる標準化')
print(xss_sk)
print(yss_sk)



#min-max正規化
mscaler = preprocessing.MinMaxScaler()
mscaler.fit(x_fit)
xms = mscaler.transform(x_fit)
mscaler.fit(y_fit)
yms = mscaler.transform(y_fit)

print('scikit-learnによるmin-max正規化')
print(xms)
print(yms)

#####################################################################重回帰分析
#標準化を使ったScikit-learn重回帰分析
model_lr_std = LinearRegression()
model_lr_std.fit(xss_sk, yss_sk)

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(model_lr_std.coef_)
print(model_lr_std.intercept_)
print(model_lr_std.score(xss_sk, yss_sk))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')

print(x_np.mean())
print(y_np.mean())
print(y_np.std())
print(x_np.std(ddof=0))
print(y_np.std(ddof=0))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(x_np.std(ddof=0))
print(y_np.std(ddof=0))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')

model_lr_std.predict(xss_sk)




#min-max正規化 Scikit-learn重回帰分析
model_lr_norm = LinearRegression()
model_lr_norm.fit(xms, yms)

print(model_lr_norm.coef_)
print(model_lr_norm.intercept_)
print(model_lr_norm.score(xms, yms))

#重回帰での偏回帰係数確認
from numpy import linalg as LA
print(LA.inv(xss_sk.T @ xss_sk) @ xss_sk.T @ yss_sk)

#決定係数R
##Sall
s_all = ((yss_sk - yss_sk.mean())**2).sum()
print(s_all)
##Sreg
s_reg = ((model_lr_std.predict(xss_sk) - yss_sk.mean())**2).sum()
print(s_reg)
##Sres
s_res = ((yss_sk - model_lr_std.predict(xss_sk))**2).sum()
print(s_res)
print('Sall: %.3f' %s_all)
print('Sreg + Sres: %.3f' %(s_reg + s_res))

#Rf
Rf = 1 - (s_res / (yss_sk.size - 4 - 1)) / (s_all / (yss_sk.size -1))
print('Rf: %.3f' %Rf)


#統計的判断
import statsmodels.api as sm

x_add_const = sm.add_constant(xss_sk)
model_sm = sm.OLS(yss_sk, x_add_const).fit()
print(model_sm.summary())

print(model_sm.aic)
