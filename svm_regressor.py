from sklearn import svm

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]

clf = svm.SVR()#regressor
clf.fit(X, y)
print(clf.predict([[1, 1]]))


#parameters tuning
#“kernel”, “gamma” and “C”.
#kernel values are:
#1) linear:(<x,x'>)
#2) polinomial:(γ<x,x'>+r)^d d is degree, r is coef0, γ is gamma>0
#3) rbf:(-gamma|x-x'|^2)) γ is gamma>0
#4) sigmoid:(tanh(γ<x,x'>+r))) r is coef0, γ is gamma>0