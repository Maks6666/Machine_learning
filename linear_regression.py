import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# to open student-mat.csv using pandas

# sep=";" - to separate data by ;
data = pd.read_csv("ML/student-mat.csv", sep=";")
# By default head() prints the first 5 lines, but you can specify the number of lines in parentheses
# for example data.head(10) to output the first 10 rows.
print(data.head())
# This line of code is for creating a new DataFrame,
# including only certain columns (features)
# from the original DataFrame data. Let's look at what these columns mean:
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# now we display the first five lines but with the given characteristics
print(data.head())
# denote a label (label) - what we want to get as a result of the prediction
predict = 'G3'

# create two arrays for attributes (dataset objects) and labels

# attributes
# remove 'G3' from attributes, which is a label
# 1 - means axis, array dimension (in this case - 1d)
x = np.array(data.drop([predict], axis = 1))


# label initialization
y = np.array(data[predict])

# divide the data into training and test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
# model accuracy
# the code literally calculates the probability with which it can guess what score it will have
# student next year
acc = linear.score(x_test, y_test)
print(f"Accuracy is {acc *100}% ")



# This code prints the coefficients and intercept
# for linear regression model.

# linear.coef_ returns the slope coefficients (weights) for each feature.
# If you have, for example, the requirement y = mx + b, then linear.coef_ contains the values
# m for each feature. Each linear.coef_ element corresponds to
# corresponding feature in your data.
# this code displays five coefficients - weights for each deviation from the schedule
# there are five of them because we work with five values ​​since data.head() is by default
# outputs only five lines

# linear.intercept_ returns the intercept term b in a linear regression equation.
# This value represents the poin where the line intersects the y-axis if all features are zero.
# In other words, this is the point where the linear regression
#  line intersects the y-axis and where x is 0
# formula y = x^n*m^n + b


print("Coef (M): \n", linear.coef_ )
print("Intercept (B): \n", linear.intercept_)


# example of a prediction for a specific student
prediction = linear.predict(x_test)

for i in range(len(prediction)):
    print(prediction[i], x_test[i], y_test[i])

# so the output would be:
# First line of output:
# 12.221295904275598 [10 9 2 0 75] 9
#12.22... is the predicted grade (G3 predicted rating) for the student.
# [10 9 2 0 75] are the input features of this student (values ​​of features G1, G2, studytime, failures, absences).
#9 is the student's actual ranking (G3) that we are trying to predict.
