from sklearn import svm

X = [[0, 0], [1, 1]]
y = [5, 8]

clf = svm.SVC()#classifier
clf.fit(X, y)
print(clf.predict([[2., 2.]]))