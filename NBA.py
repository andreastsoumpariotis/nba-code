#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:28:17 2020

@author: andreastsoumpariotis
"""
# Import packages
import os
import itertools
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import statistics
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tnrange, tqdm_notebook
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

# Reading the data
nba = pd.read_csv('NBA.csv')
nba.head(3)

# Make needed adjustments to variables
nba["3P%"] = nba["3P%"]*100
nba["FG%"] = nba["FG%"]*100
nba["2P%"] = nba["2P%"]*100
nba["0-8ftFGP"] = nba["0-8ftFGP"]*100
nba["8-16ftFGP"] = nba["8-16ftFGP"]*100
nba["16-24ftFGP"] = nba["16-24ftFGP"]*100
nba["FG"] = nba["FG"]/82
nba["FGA"] = nba["FGA"]/82
nba["3P"] = nba["3P"]/82
nba["3PA"] = nba["3PA"]/82
nba["2P"] = nba["2P"]/82
nba["2PA"] = nba["2PA"]/82

# Scatterplots
x1 = nba["3P%"]
x2 = nba["16-24ftFGP"]
x3 = nba["16-24ftFGA"]
x4 = nba["3PA"]
y1 = nba["Wins"]

# Wins vs 3PT Percentage
plt.scatter(x1, y1, marker='o')
plt.xlabel("3PT Percentage")
plt.ylabel("Total Wins")

# Wins vs 16-24Ft Field Goal Percentage
plt.scatter(x2, y1, marker='o')
plt.xlabel("16-24Ft Field Goal Percentage")
plt.ylabel("Total Wins")

# Wins vs 16-24Ft Field Goals Attempted
plt.scatter(x3, y1, marker='o')
plt.xlabel("16-24Ft Field Goals Attempted")
plt.ylabel("Total Wins")

# Wins vs 3PT Attempted
plt.scatter(x4, y1, marker='o')
plt.xlabel("3PTs Attempted")
plt.ylabel("Total Wins")

# Simple Regression on Wins on each predictor
# Wins on 3P%
wins_on_3PPerc = ols("y ~ x1", nba).fit()
print(wins_on_3PPerc.summary())
# Wins on 3PA
wins_on_3PA = ols("y ~ x2", nba).fit()
print(wins_on_3PA.summary())
# Wins on 16-24ftFGPerc
wins_on_16_24ftFGP = ols("y ~ x3", nba).fit()
print(wins_on_16_24ftFGP.summary())
# Wins on 16-24ftFGA
wins_on_16_24ftFGA = ols("y ~ x4", nba).fit()
print(wins_on_16_24ftFGA.summary())

# Multiple Regression of Wins on each predictor (3P%, 3PA, 16-24ftFGP, & 16-24ftFGA)
y = nba['Wins']
x1 = nba['3P%']
x2 = nba['3PA']
x3 = nba['16-24ftFGP']
x4 = nba['16-24ftFGA']
model = ols("y ~ x1 + x2 + x3 + x4", nba).fit()
print(model.summary())

# Correlation
corr = nba.corr()
print(corr)

# Pairplot
sns.pairplot(nba)

# Best Subset Selection

pca = PCA()
x_reduced = pca.fit_transform(scale(x))

pd.DataFrame(pca.components_.T).loc[:8,:8]

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

n = len(x_reduced)
kf_10 = KFold(n_splits=10, shuffle=True, random_state=1)

regr = LinearRegression()
mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 19 principle components, adding one component at the time.
for i in np.arange(1, 20):
    score = -1*cross_val_score(regr, x_reduced[:,:i], y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
plt.plot(mse, '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Wins')
plt.xlim(xmin=-1);

regr_test = LinearRegression()
regr_test.fit(x_reduced, y)
regr_test.coef_

def fit_linear_reg(x,y):
    #Fit linear regression model and return RSS and R squared values
    model_k = linear_model.LinearRegression(fit_intercept = True)
    model_k.fit(x,y)
    RSS = mean_squared_error(y,model_k.predict(x)) * len(y)
    R_squared = model_k.score(x,y)
    return RSS, R_squared

#Initialization variables
k = 5
RSS_list, R_squared_list, feature_list = [],[], []
numb_features = []

#Looping over k = 1 to k = 5 features in X
for k in tnrange(1,len(x.columns) + 1, desc = 'Loop...'):

    #Looping over all possible combinations: from 11 choose k
    for combo in itertools.combinations(x.columns,k):
        tmp_result = fit_linear_reg(x[list(combo)],y)   #Store temp result 
        RSS_list.append(tmp_result[0])                  #Append lists
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))   

#Store in DataFrame
df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]


df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)
df.head()

fig = plt.figure(figsize = (16,6))
ax = fig.add_subplot(1, 2, 1)

ax.scatter(df.numb_features,df.RSS, alpha = .2, color = 'darkblue' )
ax.set_xlabel('# Features')
ax.set_ylabel('RSS')
ax.set_title('RSS - Best subset selection')
ax.plot(df.numb_features,df.min_RSS,color = 'r', label = 'Best subset')
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.scatter(df.numb_features,df.R_squared, alpha = .2, color = 'darkblue' )
ax.plot(df.numb_features,df.max_R_squared,color = 'r', label = 'Best subset')
ax.set_xlabel('# Features')
ax.set_ylabel('R squared')
ax.set_title('R_squared - Best subset selection')
ax.legend()

plt.show()

#Forward Variable Selection
k = 5

remaining_features = list(x.columns.values)
features = []
RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
features_list = dict()

for i in range(1,k+1):
    best_RSS = np.inf
    
    for combo in itertools.combinations(remaining_features,1):

            RSS = fit_linear_reg(x[list(combo) + features],y)   #Store temp result 

            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1] 
                best_feature = combo[0]

    #Updating variables for next loop
    features.append(best_feature)
    remaining_features.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    features_list[i] = features.copy()
    
print('Forward stepwise subset selection')
print('Number of features |', 'Features |', 'RSS')
display([(i,features_list[i], round(RSS_list[i])) for i in range(1,5)])

df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
df1['numb_features'] = df1.index

m = len(y)
p = 10
hat_sigma_squared = (1/(m - p -1)) * min(df1['RSS'])

#Computing
df1['C_p'] = (1/m) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['AIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['BIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] +  np.log(m) * df1['numb_features'] * hat_sigma_squared )
df1['R_squared_adj'] = 1 - ( (1 - df1['R_squared'])*(m-1)/(m-df1['numb_features'] -1))
df1

df1['R_squared_adj'].idxmax()
df1['R_squared_adj'].max()

variables = ['C_p', 'AIC','BIC','R_squared_adj']
fig = plt.figure(figsize = (18,6))

for i,v in enumerate(variables):
    ax = fig.add_subplot(1, 4, i+1)
    ax.plot(df1['numb_features'],df1[v], color = 'lightblue')
    ax.scatter(df1['numb_features'],df1[v], color = 'darkblue')
    if v == 'R_squared_adj':
        ax.plot(df1[v].idxmax(),df1[v].max(), marker = 'x', markersize = 20)
    else:
        ax.plot(df1[v].idxmin(),df1[v].min(), marker = 'x', markersize = 20)
    ax.set_xlabel('Number of predictors')
    ax.set_ylabel(v)

fig.suptitle('Subset selection using C_p, AIC, BIC, Adjusted R2', fontsize = 16)
plt.show()

print(df1)

# Ridge Regression

basketball = pd.DataFrame(nba.data,columns=nba.feature_names)

y = nba.Wins
x = nba.drop(['Team','Season','16-24ftFGA'], axis = 1)
alphas = 10**np.linspace(10,-2,100)*0.5
x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(x, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('coefs')

# MSE
ridge4.fit(x, y)
pd.Series(ridge4.coef_, index = x.columns)
mean_squared_error(y, ridge4.predict(x)) #1.3149896686537381

# Lasso Regression

lasso = Lasso(max_iter = 10000, normalize = True) # can change the max iteration x10 or 100
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(x_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

# MSE
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(x_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(x_train, y_train)
mean_squared_error(y_test, lasso.predict(x_test)) #0.0001577468749999861

