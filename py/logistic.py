import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss


# ROC: Receiver Operating Characteristic
def plot_roc(true, probas):
    fpr, tpr, _ = roc_curve(true, probas)
    auc = roc_auc_score(true, probas)

    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('1 - Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title(f"Area Under the ROC Curve: {round(auc, 3)}")


def make_confusion_matrix(model, X_test, y_test, threshold=0.5):
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    fraud_confusion = confusion_matrix(y_test, y_predict)

    plt.figure(dpi=80)
    sns.heatmap(
        fraud_confusion,
        cmap=plt.cm.Blues,
        annot=True,
        square=True,
        fmt='d',
        xticklabels=['Not Looking', 'Looking'],
        yticklabels=['Not Looking', 'Looking'],
    )
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
