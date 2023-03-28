import pandas as pd
from sklearn.tree import export_graphviz

from IPython.display import Image
import pydotplus
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['crime_rate_per_100000','MURDER','RAPE','ROBBERY','AGASSLT', 'BURGLRY','LARCENY','MVTHEFT','ARSON']
# load dataset
crime = pd.read_csv("Labelled 2016 Crime Data.csv",  dtype={"crime_rate_per_100000": float, "MURDER": float,"RAPE": float
                ,"ROBBERY": float, "AGASSLT": float, "BURGLRY": float,"LARCENY": float,"MVTHEFT": float,"ARSON": float})
print(crime.head())
#crime = crime.astype({'BURGLRY': float})
feature_cols = ['MURDER','RAPE','ROBBERY','AGASSLT', 'BURGLRY','LARCENY','MVTHEFT','ARSON']
X = crime[feature_cols] # Features
y = crime.Label # Target variable
#lab = preprocessing.LabelEncoder()
#y_transformed = lab.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

clf = DecisionTreeClassifier(criterion="gini", max_depth=7)

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(y_pred)

y_pred = clf.predict([[119, 200, 1778, 3609, 4995, 13791, 3543, 464], [1, 18, 28,134,379,1598,162,16],
                     [5,43,168,	269,2163,3941,274,26],[0,	4,	0,	1,	12,	14,	0,	0]])
x_pred = ['High', 'High', 'Low', 'Low']
print(y_pred)
print("Accuracy:",metrics.accuracy_score(x_pred, y_pred))