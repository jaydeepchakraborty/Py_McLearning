from sklearn import metrics
import numpy as np


y_total = [1,0,1,2, 2, 0, 2, 2, 0, 1, 2, 2, 0, 1]#labels

expected = [2, 0, 2, 2, 0, 1]
predict = [0, 0, 2, 2, 0, 2]
labels_val = np.unique(y_total)
print("Confusion matrix:-")
print(metrics.confusion_matrix(expected, predict, labels=labels_val))
print("Accuracy:- "+ str(metrics.accuracy_score(expected,predict)))
print("Precision:- "+ str(metrics.precision_score(expected,predict,average=None)))
print("Recall:- "+ str(metrics.recall_score(expected,predict,average=None)))
print("F1score:- "+ str(metrics.f1_score(expected,predict,average=None)))
#F1 = 2 * (precision * recall) / (precision + recall)