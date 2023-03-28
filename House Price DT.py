import pandas
 # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("House_Prices.csv")
df.query('city == "Seattle"', inplace=True)
print(df)
df['price'] = df['price'].astype(int)
df['price'] = df['price'].astype(str)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
            'sqft_above'
    , 'sqft_basement']

X = df[features]
y = df.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf = DecisionTreeClassifier(criterion="gini", max_depth=200)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

y_pred = clf.predict([[5, 2.5, 3650, 9050, 2, 0, 4, 5, 3370, 280]])
print(y_pred)

y_pred = clf.predict([[5, 2.5, 2820, 67518, 2, 0, 0, 3, 2820, 0]])

print(y_pred)

