from sklearn import metrics
import numpy as np
from sklearn.cross_validation import cross_val_score,train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()

X_total = iris.data #feature vectors
y_total = iris.target #labels


#The minimum number of labels for any class cannot be less than n_folds(cv).

X, X_hold_out_set, y, y_hold_out_set = train_test_split(X_total, y_total , test_size=.2, random_state=1)
 

knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X,y)
expected = y_hold_out_set
predict = knn2.predict(X_hold_out_set)
print("Confusion matrix:-")
print(metrics.confusion_matrix(expected, predict, labels=np.unique(y_total)))
print("Accuracy:- "+ str(metrics.accuracy_score(expected,predict)))
print("Precision:- "+ str(metrics.precision_score(expected,predict)))
print("Recall:- "+ str(metrics.recall_score(expected,predict)))
print("F1score:- "+ str(metrics.f1_score(expected,predict)))
#F1 = 2 * (precision * recall) / (precision + recall)




#Training set = 60%
#Validation/Test set = 20%
#held_out_set set = 20%
#Parameter Tuning with Cross Validation
#To findout best knn model we have to do this
#Now we can vary n_neighbors and find out which model will give you best mean score
#here we are tuning n_neighbors
# knn = KNeighborsClassifier(n_neighbors=3)
# scores = cross_val_score(knn, X, y, cv = 10, scoring="accuracy")
# print(scores)
# print(scores.mean())
# m_scores = cross_val_score(knn, X, y, cv = 10, scoring="mean_squared_error")
# print((-m_scores).mean())

