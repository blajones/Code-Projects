# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Python Machine Learning Assignment
# Author: Blake Jones
#Binary Classifier from hw assignment
# Activation and weights clues from Jamar Fraction
# Date: March 7th, 2022

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics

col_names = ['crime_rate_per_100000','MURDER','RAPE','ROBBERY','AGASSLT', 'BURGLRY','LARCENY','MVTHEFT','ARSON']
# load dataset
crime = pd.read_csv("Labelled 2016 Crime Data.csv",  dtype={"crime_rate_per_100000": float, "MURDER": float,"RAPE": float
                ,"ROBBERY": float, "AGASSLT": float, "BURGLRY": float,"LARCENY": float,"MVTHEFT": float,"ARSON": float})

#crime = crime.astype({'BURGLRY': float})
feature_cols = ['MURDER','RAPE','ROBBERY','AGASSLT', 'BURGLRY','LARCENY','MVTHEFT','ARSON']
X = crime[feature_cols] # Features
y = crime.Label # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state=1)

clf = RandomForestClassifier(n_estimators=200)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
print()
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

#y_pred = clf.predict([[84.2,72.9,70.6,3.9,30,0, 0, 1, 39, .274, 3.88, 3.45, 14, 8.46]])
#print(y_pred)
y_pred = clf.predict([[119, 200, 1778, 3609, 4995, 13791, 3543, 464], [1, 18, 28,134,379,1598,162,16],
                     [5,43,168,	269,2163,3941,274,26],[0,	4,	0,	1,	12,	14,	0,	0]])
x_pred = ['High', 'High', 'Low', 'Low']
print(y_pred)
print("Accuracy:",metrics.accuracy_score(x_pred, y_pred))
feature_imp = pd.Series(clf.feature_importances_, index = feature_cols).sort_values(ascending = False)
print(feature_imp)