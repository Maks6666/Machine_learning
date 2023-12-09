import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import  KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing



# K Nearest Neighbors Algorithm -
# is a machine learning technique that is used for classification and regression.
# It is based on the proximity principle: if data points are close in feature space,
# then they will have similar target values

# Select the number of nearest neighbors (K) that will be used to make the decision.
# This may be a fixed number or determined based on some criteria.

# If K=1, then the value is simply equal to the value of its nearest neighbor.
# If K > 1, then the value can be averaged or weighted by the values ​​of its K nearest neighbors.
# K cannot be an even number

data = pd.read_csv("ML/car.data")
print(data.head()) 

# converting non-numeric data to numeric data

# This object will be used to convert categorical data to numeric data.
obj = preprocessing.LabelEncoder()


# Apply the fit_transform method to all columns of the dataset
buying = obj.fit_transform(list(data["buying"]))
maint = obj.fit_transform(list(data["maint"]))
door = obj.fit_transform(list(data["door"]))
persons = obj.fit_transform(list(data["persons"]))
lug_boot = obj.fit_transform(list(data["lug_boot"]))
safety = obj.fit_transform(list(data["safety"]))
clas = obj.fit_transform(list(data["class"]))

print(buying)

# initialize what is the label, what we want to predict
prediction = "class"

# create a list for features
x = list(zip(buying, maint, door, persons, lug_boot, safety))
 

# create a list for labels
y = list(clas)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#  reminder
# x_train: These are the training features that will be used to train the model.
# x_test: These are the test features on which we will check the operation of the model.

# y_train: These are the corresponding training set labels that are used to train the model.
# y_test: These are the corresponding test case labels that are used to test the performance of the model.




# print(x_train, y_test)

# создание классификатора 

# 5 - "K" parametr
# as higher "K" is as higher we get 
model = KNeighborsClassifier(9)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

# print(acc)


# based on test features x_test we get label prediction
predicted = model.predict(x_test)
# content of the "class" column for classification
names = ["unacc", "acc", "good", "vgood"]
# These labels (class names) are used to interpret the model results and produce predicted classes
# to a more understandable form when displaying results or analyzing the model's performance.

for i in range(len(predicted)):

# names[predicted[i]] - to link predictions to the names list
    print("Predicted: ", names[predicted[i]], "Data: ", x_test[i], "Actual: ", x_test[i])
    history = model.kneighbors([x_test[i]], 9, True)
   
    # True: This is a flag that specifies whether distances to neighbors along with their indices should be returned.
    # If True, then the method will return both indices and distances.
    
    # actually the code displays the value of the distances to all neighbors from the starting point
    # print(f"History: {history} ")