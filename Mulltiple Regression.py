from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas
import numpy as np

df = pandas.read_csv("House_Prices.csv")

df.query('city == "Seattle"', inplace= True)
print(df.head())


X = df[['bedrooms', 'bathrooms']]
y = df['price']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predicted_price = regr.predict([[3, 2]])

print(predicted_price)

print("Regression Coefficients:", regr.coef_)

X = df[['bedrooms', 'bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above'
    ,'sqft_basement']]
y = df['price']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predicted_price = regr.predict([[3, 2, 2500, 10000, 2, 0, 1, 4, 1250, 600]])

print(predicted_price)
regr.coef_ = regr.coef_.astype(int)
print("Regression Coefficients:", (regr.coef_))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

LR = LinearRegression()
LR.fit(x_train,y_train)
y_prediction =  LR.predict(x_test)
predicted_price = LR.predict([[3, 2, 2500, 10000, 2, 0, 1, 4, 1250, 600]])

print("LR Prediction", predicted_price)
score=r2_score(y_test,y_prediction)
print("r2 score is ",score)
print("mean_sqrd_error is==",mean_squared_error(y_test,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_prediction)))