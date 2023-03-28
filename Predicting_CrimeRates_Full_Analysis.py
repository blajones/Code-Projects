from urllib.request import urlopen
import json
#Getting FIP IDs of counties from web
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

#importing data file
import pandas as pd
df = pd.read_csv("2016 Crime Data.csv",
                   dtype={"FIPS": str})
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#creating crime data choropleth
fig = px.choropleth(df, geojson=counties, locations='FIPS', color='crime_rate_per_100000',
                           color_continuous_scale="Oranges",
                           scope="usa",
                           hover_name = 'county_name',
                           hover_data = ['MURDER', 'RAPE', 'ROBBERY', 'AGASSLT', 'BURGLRY'],
                           labels={'FIPS':'county_ID','AGASSLT':'ASSAULT','BURGLRY':'BURGLARY',
                                   'crime_rate_per_100000':'Crime Rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

print(df.head())

#Deleting columns from the data frame
df = df.loc[:, ~df.columns.isin(['county_name','index', 'EDITION', 'PART', 'IDNO',
                                 'FIPS_ST', 'FIPS_CTY', 'FIPS'])]
print(df)

#Creating correlation matrix
df_corr = df.corr()
print(df_corr)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = df_corr.columns,
        y = df_corr.index,
        z = np.array(df_corr),
        text=df_corr.values,
        texttemplate='%{text:.2f}'
    )
)
fig.show()

#Creating scatterplot matrix
fig = px.scatter_matrix(df)
fig.update_layout(
    font=dict(
        family="Courier New, monospace",
        size=8,
        color="RebeccaPurple"
    ))
fig.show()

#Making covariance matrix
covMatrix = df.cov()
print('Covariance Matrix:\n',covMatrix)

sns.heatmap(covMatrix, annot=True, fmt='g')
plt.show()

#Making histogram of crime rates
hist = df.crime_rate_per_100000
fig = px.histogram(hist, x="crime_rate_per_100000")
fig.show()

#First box and whisker plot
fig = go.Figure()
fig.add_trace(go.Box(
    y= df.iloc[:, 0],
    name='Mean & SD',
    marker_color='royalblue',
    boxmean='sd'
))

fig.show()

#Narrowing data to within 2.5 standard deviations
from scipy import stats
df = df[(np.abs(stats.zscore(df['crime_rate_per_100000'])) < 2.5)]
df = df.sample(frac= 1)
print(df)

#splitting data into response variable and other variables
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

#second box and whisker plot
fig = go.Figure()
fig.add_trace(go.Box(
    y= df.iloc[:, 0],
    name='Mean & SD',
    marker_color='royalblue',
    boxmean='sd' # represent mean and standard deviation
))

fig.show()

#splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state= 1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Doing PCA
from sklearn.decomposition import PCA

pca = PCA()
xtrain = pca.fit_transform(X_train)
xtest = pca.transform(X_test)

#Output of explained variance by PCA
explained_variance = pca.explained_variance_ratio_

print(explained_variance)
#How much each variable contributes to the PCA
print(abs( pca.components_ ))

#Line graph of explained variance
df2 = pd.DataFrame(dict(
    x = list(range(16)),
    y = explained_variance
))

df2 = df2.sort_values(by="y")
fig = px.line(df2, x="x", y="y", title="Variance")
fig.show()

from sklearn.decomposition import PCA
#Using PCA of 4 for regression models
pca = PCA(n_components= 4, random_state= 1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

#Linear regression model
model = LinearRegression()
model.fit(xtrain, y_train)
score = model.score(xtrain, y_train)
print("R-squared:", score)
print('Adj_R^2',1 - (1-model.score(xtrain, y_train))*(len(y_train)-1)/(len(y_train)-xtrain.shape[1]-1))
predictions = model.predict(xtest)
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

params = np.append(model.intercept_,model.coef_)
newX = pd.DataFrame({"Constant":np.ones(len(xtrain))}).join(pd.DataFrame(xtrain))
#newX = pd.DataFrame({"Constant":np.ones(len(xtrain))}).join(pd.DataFrame(xtrain.reset_index(drop=True)))
MSE = (sum((y_test -predictions)**2))/(len(newX)-len(newX.columns))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)- 1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
print(myDF3)

#Random forest regression model

rfr = RandomForestRegressor(n_estimators= 500, max_depth=100, random_state= 1)
rfr.fit(xtrain, y_train)

score = rfr.score(xtrain, y_train)
print("R-squared:", score)
print('Adj_R^2',1 - (1-rfr.score(xtrain, y_train))*(len(y_train)-1)/(len(y_train)-xtrain.shape[1]-1))
ypred = rfr.predict(xtest)
fmse = mean_squared_error(y_test, ypred)
mse = mean_squared_error(y_test, ypred, squared= False)
print("MSE: ", fmse)
print("Squared MSE: ", mse)
print('mean_absolute_error : ', mean_absolute_error(y_test, ypred))


#Created after progress report
#Using xgboost regression models
import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective ='reg:pseudohubererror', colsample_bytree = 0.8, learning_rate = 0.25,
                max_depth = 200, alpha = 5, n_estimators = 500, seed=1)
xg_reg.fit(xtrain,y_train)
score = xg_reg.score(xtrain, y_train)
print("R-squared:", score)
print('Adj_R^2',1 - (1-xg_reg.score(xtrain, y_train))*(len(y_train)-1)/(len(y_train)-xtrain.shape[1]-1))
ypred = xg_reg.predict(xtest)
fmse = mean_squared_error(y_test, ypred)
mse = mean_squared_error(y_test, ypred, squared= False)
print("MSE: ", fmse)
print("Squared MSE: ", mse)
print('mean_absolute_error : ', mean_absolute_error(y_test, ypred))


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.8, learning_rate = 0.25,
                max_depth = 200, alpha = 5, n_estimators = 500, seed=1)
xg_reg.fit(xtrain,y_train)
score = xg_reg.score(xtrain, y_train)
print("R-squared:", score)
print('Adj_R^2',1 - (1-xg_reg.score(xtrain, y_train))*(len(y_train)-1)/(len(y_train)-xtrain.shape[1]-1))
ypred = xg_reg.predict(xtest)
fmse = mean_squared_error(y_test, ypred)
mse = mean_squared_error(y_test, ypred, squared= False)
print("MSE: ", fmse)
print("Squared MSE: ", mse)
print('mean_absolute_error : ', mean_absolute_error(y_test, ypred))


#Lightgbm regression
import lightgbm as ltb
model = ltb.LGBMRegressor(boosting_type = 'dart', learning_rate = 0.2, max_depth = 0,
                          n_estimators=500, num_leaves = 200, random_state= 1 )
model.fit(xtrain,y_train)
score = model.score(xtrain, y_train)
print("R-squared:", score)
print('Adj_R^2',1 - (1-model.score(xtrain, y_train))*(len(y_train)-1)/(len(y_train)-xtrain.shape[1]-1))
ypred = model.predict(xtest)
fmse = mean_squared_error(y_test, ypred)
mse = mean_squared_error(y_test, ypred, squared= False)
print("MSE: ", fmse)
print("Squared MSE: ", mse)
print('mean_absolute_error : ', mean_absolute_error(y_test, ypred))

