from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn import svm, datasets
import math
import itertools



actual = [2, 0, 2, 2, 0, 2]#y_test
predictions = [0, 0, 2, 2, 0, 2]#y_pred


######################## Print Evaluation Metirces start ###########################
def prntEvalMetrices():
    print("Accuracy:- "+ str(metrics.accuracy_score(actual,predictions)))
    print("Precision:- "+ str(metrics.precision_score(actual,predictions,average=None)))
    print("Recall:- "+ str(metrics.recall_score(actual,predictions,average=None)))
    print("F1score:- "+ str(metrics.f1_score(actual,predictions,average=None)))
    
    print("Kappa:- "+ str(metrics.cohen_kappa_score(actual,predictions)))
######################## Print Evaluation Metirces End ###########################



######################## plotting confusion matrix start ###########################
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
######################## plotting confusion matrix End ###########################


######################## print confusion matrix Start ###########################
def prntConfMtrix():

    # Compute confusion matrix
    cnf_matrix = metrics.confusion_matrix(actual, predictions)
    print(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[0,2],title='Confusion matrix, without normalization')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[0,2], normalize=True,title='Normalized confusion matrix')
    
    plt.show()
######################## print confusion matrix End ###########################

######################## plot ROC curves for multi-class problem Start ###########################
def pltMulClsROC():
    
    
    y = np.array([1, 1, 2, 2])
    scores = np.array([0.1, 0.4, 0.35, 0.8])# prob score of output
    
    
    classes = [1,2] #it is unique number of classes
    n_classes = 2
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        #pos_label : int or str, default=None
        #Label considered as positive and others are considered negative.
        # false positive rates
        # true positive rates
        fpr[i], tpr[i], _ = roc_curve(y, scores, pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    lw =2#line width
    # Compute macro-average ROC curve and ROC area
    
    
    # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(actual, predictions)
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle=':', linewidth=4)
    
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["macro"]),
#              color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
######################## plot ROC curves for multi-class problem End ###########################



######################## Printing R-Squared and Mean Squared Error Start ###########################
def prntMSE():
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print("R squared :- "+ str(r2_score(y_true, y_pred)))
    print("Mean squared error:- "+ str(mean_squared_error(y_true, y_pred)))
######################## Printing R-Squared and Mean Squared Error End ###########################

######################## Printing Log Loss Start ###########################
def prntLogLoss():
    #Suppose you have three classes "spam", "ham", "bam"
     
    y_true = ["bam","ham","spam","spam"] #So we have 4 rows and their actual labels
    y_pred = [[0.8, 0.1, 0.2], #here first row has probs of three different class labels, 
               [0.3, 0.6, 0.1], #probs are ordered alphabetically, so for first row
               [0.15, 0.15, 0.7], #0.8 is for bam, 0.1 is for ham and 0.2 is for spam
               [0.05, 0.05, 0.9]]
     
    ll  = log_loss(y_true, y_pred)
    
    print("Log Loss :- "+str(ll))
    print("Accurcy is :- " + str (math.exp(-ll)))

######################## Printing Log Loss End ###########################


def mainMtdh():
    #prntEvalMetrices()
    #prntConfMtrix()
    pltMulClsROC()
    #prntMSE()
    #prntLogLoss()
    
    
mainMtdh()    
    
    
