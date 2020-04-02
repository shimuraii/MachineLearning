import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "absences", "freetime"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# section of array, section of y array
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)


"""
best = 0
for g in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
# test removes about 10 % of data for testing, (0.1 test_size)
#linear regression is the best fit line of a 3 dimensional graph

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train) #fits the data to get a best fit line
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in) # loads linear graph

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
# prediction [ rates ] actual grade
p = "freetime"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
