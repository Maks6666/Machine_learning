import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import pickle
from matplotlib import style


data = pd.read_csv("ML/student-mat.csv", sep=";")
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())
predict = 'G3'
x = np.array(data.drop([predict], axis = 1))


y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# model save

# for i in range(30): - The loop is repeated 30 times.
# linear = linear_model.LinearRegression() - Creates a new linear regression model object.
# linear.fit(x_train, y_train) - The model is trained on the training data (x_train, y_train).
# Model accuracy assessment:

# acc = linear.score(x_test, y_test) - Calculate the accuracy of the model on test data (x_test, y_test).
# Saving the model:

# if acc > best: - If the current accuracy exceeds the best previously stored accuracy, then:
# best = acc - Best accuracy is updated.
# with open("studentmodel.pickle", "wb") as f:
# - Opens the file "studentmodel.pickle" to write bytes in binary writing mode ("wb").
# pickle.dump(linear, f) - Saves the current model to a file.
# Loading the best model:

# pickle_in = open("studentmodel.pickle", "rb")
# - Opens the file "studentmodel.pickle" to read bytes in binary read mode ("rb").
# linear = pickle.load(pickle_in) - Loads a model object from a file.
# So, as a result of executing the code, you will have a linear variable,
# containing the best linear regression model out of 30 trained models,
# and also this model will be saved in the file "studentmodel.pickle".
# You can use this model to predict scores
# students or for further analyses.


"""
#
# 

best = 0
for i in range(30):

    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    # код буквально расчитывает вероятность с которой он может угадать какая оценка будет у 
    # студента в следующем году
    acc = linear.score(x_test, y_test)
    print(f"Accuracy is {acc}% ")
    if acc > best:
        best = acc



        # to save model 
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coef (M): \n", linear.coef_ )
print("Intercept (B): \n", linear.intercept_)

prediction = linear.predict(x_test)

for i in range(len(prediction)):
    print(prediction[i], x_test[i], y_test[i])



# plotting

# p = "G1" - Defines the variable p as the string "G1".
# This is used to select the corresponding data column from the DataFrame data.

# you can specify any attribute
p = "G1"

# - Sets the plot style, in this case, "ggplot". This is one of the styles
# provided by Matplotlib, which mimics the style of plots in the ggplot package
# in the R programming language.
style.use("ggplot")

# Creates a scatter plot where data[p] represents the x-axis values,
# and data[predict] are the values ​​along the Y axis. In this case, it is assumed that data[p]
# are the values ​​of the variable "G1" (for example, the first estimate),
# and data[predict] are the values ​​we want to predict
# (eg "G3" or final result).
plt.scatter(data[p], data[predict])

# x-axis name
plt.xlabel(p)

# y-axis name
plt.ylabel("Final result")

# e
plt.show()


# In this graph, plt.scatter(data[p], data[predict]), the X axis will represent the values ​​of the variable "G1"
# (student's first grade), and the Y axis is the value of the variable "G3"
# (student's final grade or the result we are trying to predict).
# Each point on the graph will represent a pair of values ​​"G1" and "G3"
# for each student in your dataset.

# So if a student has a "G1" of, say, 10, and a "G3" of 15,
# then the point on the graph will be at the level X=10 and Y=15.
# The scatter plot allows you to visually assess
# is there any relationship between the student's initial and final grade.

# After analyzing the graph, you can conclude that
# how much the variable "G1" affects the variable "G3" and how
# are there any obvious trends or patterns in this data.