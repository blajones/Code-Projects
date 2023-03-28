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

col_names = ['Temperature','Dew Point','Humidity','Wind Speed','Pressure' ,'Precipitation','Arm_N','Stadium_N','Pitcher_N',
             'AVG','OBP','PERA', 'TERA','OPP', 'HPN']

bat = pd.read_csv("Altuve_BattinInfo.csv",  dtype={"Temperature": float, "Dew Pont": float,"Humidity": float
                ,"Wind Speed": float, "Pressure": float, "Precipitation": float,"Arm_N": float,"Stadium_N": float,
                 "Pitcher_N": float, "AVG": float, "OBP": float, "SLG": float, "OPS": float})

feature_cols = ['Temperature','Dew Point','Humidity','Wind Speed','Pressure' ,'Precipitation','Arm_N',
                'Stadium_N','Pitcher_N', 'AVG','PERA', 'TERA','OPP', 'HPN']
X = bat[feature_cols] # Features
y = bat.Hit_Label # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

clf = RandomForestClassifier(n_estimators=100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
print()
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

y_pred = clf.predict([[84.2,72.9,70.6,3.9,30,0, 0, 1, 39, .274, 3.88, 3.45, 14, 8.46]])
print(y_pred)

feature_imp = pd.Series(clf.feature_importances_, index = feature_cols).sort_values(ascending = False)
print(feature_imp)