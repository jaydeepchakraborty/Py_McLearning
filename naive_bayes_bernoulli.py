import numpy as np
from sklearn.naive_bayes import BernoulliNB

#classifier
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 4, 4, 4])
clf = BernoulliNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))