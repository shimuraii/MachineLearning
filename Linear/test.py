import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# section of array, section of y array
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
# test removes about 10 % of data for testing, (0.1 test_size)
#linear regression is the best fit line of a 3 dimensional graph

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train) #fits the data to get a best fit line
acc = linear.score(x_test, y_test)
print(acc)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# prediction [ rates ] actual grade
