import pandas
from keras.models import Sequential
from keras.layers import Dense


df = pandas.read_csv("House_Prices_2.csv")

df.query('city == "Seattle"', inplace= True)
print(df)

feature_cols = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above'
    ,'sqft_basement']

X = df[feature_cols]
y = (df['High'])

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(10,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
