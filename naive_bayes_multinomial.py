#The multinomial Naive Bayes classifier is suitable for classification with discrete features 
#(e.g., word counts for text classification). The multinomial distribution normally requires 
#integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

import numpy as np
from sklearn.naive_bayes import MultinomialNB

#classifier
X = np.array([[1, 1], [2, 1], [3, 2]])
Y = np.array([1, 5, 5])
clf = MultinomialNB()
clf.fit(X, Y)
print(clf.predict([[0.8, 1]]))