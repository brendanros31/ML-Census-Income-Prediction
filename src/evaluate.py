import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics



# Prediction Stats
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)   # Predictions
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()   # Positives/Negetives
    
    # Confusion matrix
    print(model)
    print(f'\n{metrics.confusion_matrix(y_test, y_pred)}\n')

    # Classification report
    print(f'Classification report: \n{metrics.classification_report(y_test, y_pred)}\n')

    # Specificity
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.2f}\n")

    # Total Support
    total_sup = tn+fp+fn+tp
    print(f"Total support: {total_sup:.2f}\n")




# ROC score, curve
def roc(model, X_test, y_test):
    y_pred = model.predict(X_test)   # Predictions

    logit_roc_auc = metrics.roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,-1])

    # Plot
    plt.figure(figsize=(12,6))

    plt.title('Reciever Operating Characteristic')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.xlim([0.0, 1.0])  # Limits for the x-axis
    plt.ylim([0.0, 1.05])  # Limits for the y-axis

    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)  
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line for reference

    plt.legend(loc='lower right')

    # plt.savefig('Log_ROC')
    plt.show()
