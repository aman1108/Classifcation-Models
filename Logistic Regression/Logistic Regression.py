import pandas as pd
import numpy as np
from sklearn import preprocessing 
import matplotlib.pyplot as plt

import pylab as pl
import scipy.optimize as opt
from sklearn.metrics import confusion_matrix

churn_df = pd.read_csv("ChurnData.csv")

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
#print(churn_df.head())

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5])
y = np.asarray(churn_df['churn'])
print(y[0:5])

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.linear_model import LogisticRegression
print(X_train,y_train)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
#print(LR)

yhat = LR.predict(X_test)
#print(yhat)
yhat_prob = LR.predict_proba(X_test)
#print(yhat_prob)

from sklearn.metrics import jaccard_score
#print(jaccard_score(y_test, yhat))

#COnfusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
#plt.show()



#direct using confusion matrix
#print(confusion_matrix(y_test, yhat, labels=[1,0]))
#print (classification_report(y_test, yhat))


from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))

