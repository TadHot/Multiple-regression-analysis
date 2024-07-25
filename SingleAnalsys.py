#単回帰分析　
#https://qiita.com/karaage0703/items/701367b6c926552fe505  参考サイト

import pandas as pd
df = pd.read_excel('BostonDataSet.xlsx', sheet_name=0, header=0)
#df = pd.read_excel('test_data.xlsx', sheet_name=0, header=0)
matrix = df.shape #行数、列数取得
print(matrix)

#CRIM   ZN  INDUS  CHAS   NOX      RM   AGE     DIS  RAD  TAX  PTRATIO  MEDV 
"""
 0 CRIM     per capita crime rate by town
 1 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 2 INDUS    proportion of non-retail business acres per town
 3 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 4 NOX      nitric oxides concentration (parts per 10 million)
 5 RM       average number of rooms per dwelling
 6 AGE      proportion of owner-occupied units built prior to 1940
 7 DIS      weighted distances to five Boston employment centres
 8 RAD      index of accessibility to radial highways
 9 TAX      full-value property-tax rate per $10,000
 10 PTRATIO  pupil-teacher ratio by town
 11 MEDV     Median value of owner-occupied homes in $1000's
 """


head = df.columns[0:11] #11
print('HEAD: ',head)

l1=int((input('Enter the first value: '))) #番号で入力
pattern = head[l1]
print(pattern)

""" データを列毎に取得　"""
MEDV_d = df.loc[:,'MEDV']  #住宅価格
x_d = df.loc[:,pattern]
"""CRIM_d = df.loc[:,'CRIM']   #列取得
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

#先ずは単回帰分析　y=MEDV x=上のどれか
import matplotlib.pyplot as plt     #グラフ化
import seaborn as sns

plt.plot(x_d, MEDV_d, 'o')                           ####データ変更
#plt.show()

import numpy as np    #numpyで分析
np_x = x_d.values                                     ####データ変更
np_y = MEDV_d.values
np_xy = np.stack([np_x, np_y])

"""print('np_x:', np_x)
print('np_y:', np_y)
print('np_xy:')
print(np_xy)
"""

#scikit-learn
from sklearn.linear_model import LinearRegression

x=np.array(np_x).reshape(-1,1)
y=np.array(np_y).reshape(-1,1)
#print(x)
#print(y)

model_lr = LinearRegression()
model_lr.fit(x, y)

plt.plot(x_d, MEDV_d, 'o')                             ####データ変更
plt.plot(x, model_lr.predict(x), linestyle="solid")
plt.show()

print('モデル関数の回帰変数 w1: %.3f' %model_lr.coef_)
print('モデル関数の切片 w2: %.3f' %model_lr.intercept_)
print('y= %.3fx + %.3f' % (model_lr.coef_ , model_lr.intercept_))
print('決定係数 R^2： ', model_lr.score(x, y))

#分散、共分散
s_xy = np.cov(np_xy, rowvar=1, bias=1)
print(s_xy)

s_xx = np.var(np_x)
s_yy = np.var(np_y)
#print('S_xx : %.3f' %s_xx)
#print('S_yy : %.3f' %s_yy)

#回帰係数w1
w_1 = s_xy[0][1] / s_xx
print('w1 : %.3f' %w_1)
#回帰係数w0
np_x_mean = np_x.mean()
np_y_mean = np_y.mean()
w_0 = np_y_mean - w_1 * np_x_mean
print('w0 : %.3f' %w_0)

#決定変数Rの算出
##全変動Sall
s_all = ((MEDV_d.values - MEDV_d.values.mean())**2).sum()
print(s_all)
##Sreg
s_reg = ((model_lr.predict(x) - y.mean())**2).sum()
print(s_reg)
##Sres
s_res = ((y - model_lr.predict(x))**2).sum()
print(s_res)
#R^2
R2 = s_reg / s_all
print('R^2: %.3f' %R2)
#R^2
r = s_xy[0][1] / (s_xx * s_yy)**(0.5)
r2 = r**2
print('r2: %.3f' %r2)
print('R^2: %.3f' %R2)

#
import statsmodels.api as sm

x_add_const = sm.add_constant(x)
model_sm = sm.OLS(y, x_add_const).fit()
print(model_sm.summary())