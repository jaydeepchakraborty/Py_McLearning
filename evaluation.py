from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, datasets

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

actual = [2, 0, 2, 2, 0, 2]
predictions = [0, 0, 2, 2, 0, 2]


print("Confusion matrix:-")
# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(actual, predictions)
print(cnf_matrix)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,2],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,2], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
print("Accuracy:- "+ str(metrics.accuracy_score(actual,predictions)))
print("Precision:- "+ str(metrics.precision_score(actual,predictions,average=None)))
print("Recall:- "+ str(metrics.recall_score(actual,predictions,average=None)))
print("F1score:- "+ str(metrics.f1_score(actual,predictions,average=None)))

#precision=TP / (TP + FP) 98.16%
#sensitivity/recall = TP / (TP + FN) 98.55%
#specificity = TN / (FP + TN) 99.61% not available in python
#F-score = 2*TP /(2*TP + FP + FN) 98.35%


print("Kappa:- "+ str(metrics.cohen_kappa_score(actual,predictions)))

#pos_label : int or str, default=None
#Label considered as positive and others are considered negative.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=2)
# false positive rates
#true positive rates
metrics.auc(fpr, tpr)






from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
r2_score(y_true, y_pred)
mean_squared_error(y_true, y_pred)