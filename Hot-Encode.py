import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("House_Prices_2.csv")


df = pd.get_dummies(df, columns = ['city'])
print(df)

df = df.loc[:, ~df.columns.isin(['date', 'yr_built','yr_renovated', 'statezip','country', 'street', 'city', 'Low'])]

print(df)

df = df.astype(int)
print(df.loc[:, df.columns != "price"])
X = df.loc[:, df.columns != "price"]
y = np.ravel(df['High'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(55,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer='sgd',
metrics=['accuracy'])

model.fit(X_train, y_train,epochs=100, verbose=1)

y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test,verbose=1)
print(score)
