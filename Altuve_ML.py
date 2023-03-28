# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Python Apriori Algorithm Assignment
# Author: Blake Jones
# Date: Feb 12th, 2022

# Itertools from stackoverflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['Temperature','Dew Point','Humidity','Wind Speed','Pressure' ,'Precipitation','Arm_N','Stadium_N','Pitcher_N',
             'AVG','OBP','PERA', 'TERA','OPP', 'HPN']

bat = pd.read_csv("Altuve_BattinInfo.csv",  dtype={"Temperature": float, "Dew Pont": float,"Humidity": float
                ,"Wind Speed": float, "Pressure": float, "Precipitation": float,"Arm_N": float,"Stadium_N": float,
                 "Pitcher_N": float, "AVG": float, "OBP": float, "SLG": float, "OPS": float})

feature_cols = ['Temperature','Dew Point','Humidity','Wind Speed','Pressure' ,'Precipitation','Arm_N',
                'Stadium_N','Pitcher_N', 'AVG','PERA', 'TERA','OPP', 'HPN']
X = bat[feature_cols] # Features
y = bat.Hit_Label # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 85% training

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#y_pred = clf.predict([[86.4, 67.3, 55.5, 22.9, 29.2, 0, 1, 10, 38, .279]])
#print(y_pred)
i = 0
while i <11:
    clf = DecisionTreeClassifier(criterion="gini", max_depth=7)

    clf = clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print(y_pred)

    y_pred = clf.predict([[84.2,72.9,70.6,3.9,30,0, 0, 1, 39, .274, 3.88, 3.45, 14, 8.46]])
    print(y_pred)
    i += 1
#x_pred = ['Yes']
#print("Accuracy:",metrics.accuracy_score(x_pred, y_pred),)     
