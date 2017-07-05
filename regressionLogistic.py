from sklearn import linear_model, datasets
import numpy as np

#=====================================================================================
#regressor
# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LogisticRegression(C=1e5)
regr.fit(diabetes_X_train, diabetes_y_train)

print(regr.predict([[-0.01590626]]))

# The mean squared error
mse = np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
print("Mean squared error: %.2f" %mse )

## Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

#======================================================================================
#regressor

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

#print(X)
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
print(logreg.predict([[ 14, 5]]))