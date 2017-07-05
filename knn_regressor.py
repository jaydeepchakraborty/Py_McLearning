from sklearn import metrics
from sklearn.cross_validation import cross_val_score,train_test_split
from sklearn.neighbors import KNeighborsRegressor



X = [[0], [1], [2], [3]]
y = [285, 300, 456, 148]
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X, y) 
print(knn.kneighbors([[1.5]], n_neighbors=2, return_distance=True))
#(array([[ 0.5,  0.5]]), array([[1, 2]]))
#means array([[1, 2]] ==> [1], [2] are nearest neighbors
#and their distances are array([[ 0.5,  0.5]])
print(knn.predict([[1.5]]))
#prediction will be average of neighbors
#[1] ==> 300 , [2] ==> 456
#(300+456)/2 is 378
#[ 378.]





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

