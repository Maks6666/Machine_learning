
# svm (supprot vector machine) - a method based on searching for the optimal separating hyperplane (separating vector)
# the most imaginative one is the one that forms the widest line


import sklearn 
from sklearn import datasets 
from sklearn import svm 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# dataset uploading 
data = datasets.load_breast_cancer()

# Printing feature names
print(data.feature_names)

# Print class names (target variables)
print(data.target_names)

# features
x = data.data 


# target vector - what the model will predict
y = data.target 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
print(x_train, y_train)


# classes into which we will classify cancerous tumors
classes = ['malignant' 'benign']


# using the kernel method to convert a graph from two-dimensional to three-dimensional, in addition to all this, increases the classification accuracy


# in this case the kernel has a linear parameter.
# Linear kernel parameter means SVM will look for linear separating hyperplane
# in the original feature space.
classifier = svm.SVC(kernel="linear")

# the "linear" parameter also has its own C parameter
# it is a regularization parameter.
# This parameter controls the balance between achieving the optimal bound
# separation between classes and minimizing classification errors.
# also promotes high accuracy
# classifier = svm.SVC(kernel="linear", C = 2)

# you can also select the polyum kernel parameter
# A polynomial kernel can be useful when the data cannot be effectively divided by a straight line.
# It adds nonlinearity to the feature space,
# which allows the model to build more complex separating hyperplanes,
# adapting to non-linear dependencies in the data. Less effective in this case
# classifier = svm.SVC(kernel="poly")

# to make this method more efficient, you can also add a degree parameter
# The degree parameter is used in the context of a polynomial kernel (kernel="poly").
# This parameter determines the degree of the polynomial that is used to transform the data
# to higher dimensional space.
# classifier = svm.SVC(kernel="poly", degree=2)

# Radial basis function (RBF or Gaussian) kernel.
# This is one of the most popular kernels in SVM. It allows the model to construct nonlinear separation boundaries,
# which is useful in cases where the data is complex and cannot be well separated by a linear hyperplane.
# in our case also has a positive effect on the accuracy of the model


# classifier = svm.SVC(kernel="rbf")

# we can also compare the knn and svm methods
# in this case their effectiveness is almost the same
# classifier = KNeighborsClassifier(n_neighbors=7)

classifier.fit(x_train, y_train)

# test sample prediction for features
prediction = classifier.predict(x_test)

# measure the accuracy of the model by comparing predicted values ​​(prediction)
# with actual values ​​(y_test) for the test sample.
accuracy = metrics.accuracy_score(y_test, prediction)

print(accuracy)

# prediction for each feature output
for i in range(len(prediction)):
    print("Predicted value:", prediction[i], "Actual value:", y_test[i])




