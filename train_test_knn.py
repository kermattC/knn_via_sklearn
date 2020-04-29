# going to try to make a classifier that can predict the make of your car. HERE WE GOOOOOOOOO
# due to limitations of the data, this is only valid for cars from 1990 to 2017 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

data_set = pd.read_csv('cars.csv')

x = data_set[['Year', 'Engine Fuel Type', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 'Driven_Wheels', 'highway MPG', 'city mpg','MSRP']]

# numerize fuel types
x = x.replace(['regular unleaded', 'premium unleaded (required)', 'premium unleaded (recommended)', 'flex-fuel (premium unleaded recommended/E85)', 'flex-fuel (unleaded/E85)', 'flex-fuel (premium unleaded required/E85)', 'flex-fuel (unleaded/natural gas)', 'natural gas', 'diesel', 'electric'], [0,1,1,2,2,2,2,2,3,4])
# numerize transmission type
# x = x.replace(['MANUAL', 'AUTOMATIC', 'UNKNOWN', 'AUTOMATED_MANUAL', 'DIRECT_DRIVE'], [0, 1, 1, 2,3])
# numerize driven wheels
# x = x.replace(['rear wheel drive', 'front wheel drive', 'four wheel drive', 'all wheel drive'],[0,1,2,3])

# our goal
y = data_set['Make'].values

# next we gotta normalize the data

# standardizing via sklearn. This gives data a zero mean and scales to unit variance
normalized_x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
# impute NaN values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(normalized_x)   # calculate impute value
normalized_x = imp.transform(normalized_x)  # apply changes to object

"""
# normalizing using pandas (thank god for pandas cuz this makes life so much easier)
if you want to see what i mean, type add this line of code and see the value:
print(x.mean())

# via mean normalization
# normalized_x = (x - x.mean()) / x.std()

# via min/max normalized data
normalized_x = (x - x.mean()) / (x.max() - x.min())

# fill in NaN's with mean
normalized_x = normalized_x.fillna(normalized_x.mean())

""" 
# now we split our data
# one method of doing this is via scikit learn. 
x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, test_size = 0.25, random_state = 69)

"""
Tried to train data via pandas but i failed

# sample takes a random fraction from the data frame. In this case, we choose 1 so we basically shuffle the entire thing
shuf_normalized_x = normalized_x.sample(frac=1, random_state = 69)
define train set size
train_size = int(0.75 * len(x))
split dataset
train_set = shuf_normalized_x[:train_size]
test_set = shuf_normalized_x[train_size:]

print("Train set: " , train_set)
print("Test  set: ", test_set)
"""

# now we make the classifier
k = 4
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(x_train, y_train)

# prediction
y_predict = classifier.predict(x_test)
# compare actual and prediction
# print(y_predict[0:10])
# print(y_test[0:10])                   

# evaluate the accuracy
# an interesting result I found was that normalizing values via sklearn and mean normalization yielded highest f1 score
#   meanwhile min/max normalization yielded the lowest f1-score. I wonder why... FIND OUT IN THE NEXT EPISODEEEE
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

# let's try to find the optimal k value. test from 1 to 20
# source code from https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    predict_i = knn.predict(x_test)
    error.append(np.mean(predict_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='green', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()