#重回帰分析　
import pandas as pd
import numpy as np    #numpy
import itertools
#import sklearn
df = pd.read_excel('BostonDataSet.xlsx', sheet_name=0, header=0)
#df = pd.read_excel('winequality-red.xlsx', sheet_name=0, header=0)

matrix = df.shape #行数、列数取得
print(matrix)
print(df)

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

#データセット
     #y=MEDV x1,x2,x=上のどれか
head = df.columns[0:11] #11
print('HEAD: ',head)
#x = df.loc[:,["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO"]]

#patterns = itertools.combinations(head, 4)
patterns = itertools.permutations(head, 4)
for pattern in patterns:
    #print(pattern)
    x = df.loc[:,pattern]
    y = df.loc[:,'MEDV']

    #print(x)
    #print(y)
    #print(x.shape)
    #print(y.shape)


    
    """nm = np.array(pattern)
    #nm1 = nm[0]
    nm2 = nm[1]
    nm3 = nm[2]
    nm4 = nm[3]

    print(nm1,nm2,nm3,nm4)


    x1 = df.loc[:,nm1]
    x2 = df.loc[:,nm2]
    x3 = df.loc[:,nm3]
    x4 = df.loc[:,nm4]"""

    #####################################################################標準化・正規化
    #重回帰分析で、入力変数が複数になったことで正規化の必要性
    #分散を用いて標準化

    #numpyによる正規化　
    #yがNaNになる
    """x_np = x.apply(lambda x: (x - np.mean(x)) / np.std(x))
    #y_np = y.apply(lambda y: (y - np.mean(y)) / np.std(y))    #yは一列しかないからapply関数はいらないのでは？これやるとNaNになる。
    y_np = (y - np.mean(y)) / np.std(y)

    print(x_np.head())
    print(y_np.head())

    #pandasによる正規化
    xss_pd = (x - x.mean()) / x.std(ddof=0)
    yss_pd = (y - y.mean()) / y.std(ddof=0)

    print(xss_pd.head())
    print(yss_pd.head())"""

    #scikit-learnによる正規化
    from sklearn import preprocessing
    from sklearn.linear_model import LinearRegression

    sscaler = preprocessing.StandardScaler()
    #x_fit =np.array(x).reshape(-1,1) #これで変換しないとfitに入力できない
    x_fit = x
    y_fit = np.array(y).reshape(-1,1) #これで変換しないとfitに入力できない

    sscaler.fit(x_fit)
    xss_sk = sscaler.transform(x_fit) 
    sscaler.fit(y_fit)
    yss_sk = sscaler.transform(y_fit)

    #print(xss_sk)
    #print(yss_sk)

    #min-max正規化
    mscaler = preprocessing.MinMaxScaler()
    mscaler.fit(x_fit)
    xms = mscaler.transform(x_fit)
    mscaler.fit(y_fit)
    yms = mscaler.transform(y_fit)

    #print(xms)
    #print(yms)

    #####################################################################重回帰分析
    #標準化を使ったScikit-learn重回帰分析
    model_lr_std = LinearRegression()
    model_lr_std.fit(xss_sk, yss_sk)

    """
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print(model_lr_std.coef_)
    print(model_lr_std.intercept_)
    print(model_lr_std.score(xss_sk, yss_sk))

    model_lr_std.predict(xss_sk)

    #min-max正規化 Scikit-learn重回帰分析
    model_lr_norm = LinearRegression()
    model_lr_norm.fit(xms, yms)
    print('+++++++++++++++++++++++++++++++++++++++++++')
    print(model_lr_norm.coef_)
    print(model_lr_norm.intercept_)
    print(model_lr_norm.score(xms, yms))

    #重回帰での偏回帰係数確認
    from numpy import linalg as LA
    print(LA.inv(xss_sk.T @ xss_sk) @ xss_sk.T @ yss_sk)
    """
    #決定係数R
    ##Sall
    s_all = ((yss_sk - yss_sk.mean())**2).sum()
    #print(s_all)
    ##Sreg
    s_reg = ((model_lr_std.predict(xss_sk) - yss_sk.mean())**2).sum()
    #print(s_reg)
    ##Sres
    s_res = ((yss_sk - model_lr_std.predict(xss_sk))**2).sum()
    #print(s_res)
    #print('Sall: %.3f' %s_all)
    #print('Sreg + Sres: %.3f' %(s_reg + s_res))

    #Rf
    Rf = 1 - (s_res / (yss_sk.size - 4 - 1)) / (s_all / (yss_sk.size -1)) #4変数だから４
    #print('Rf: %.3f' %Rf)
   

    #統計的判断
    import statsmodels.api as sm

    x_add_const = sm.add_constant(xss_sk)
    model_sm = sm.OLS(yss_sk, x_add_const).fit()
    #print(model_sm.summary())

    print(pattern,'AIC: %.1f' %model_sm.aic,'Rf: %.3f' %Rf)
