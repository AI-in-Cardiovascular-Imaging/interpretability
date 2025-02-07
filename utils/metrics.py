from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support, confusion_matrix, accuracy_score,
                             average_precision_score, ConfusionMatrixDisplay, roc_curve)
import numpy as np


def compute_metrics(y_true, y_pred, thresh):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Calculate Youden Index for each threshold
    youden_index = tpr - fpr
    # Find the threshold that maximizes the Youden Index
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    # Compute metrics
    label_pred = y_pred >= thresh
    precision, recall, f1score, _ = precision_recall_fscore_support(y_true, label_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, label_pred).ravel()
    specificity = tn / (tn + fp)

    # Store metrics into the output dictionary
    metrics = {
        "AUC": roc_auc_score(y_true, y_pred),
        "Sen": recall,
        "Spe": specificity,
        "youden_index": recall + specificity - 1,
        "f1-score": f1score,
        "ppv": precision,
        "npv": tn / (tn + fn),
        "aupr": average_precision_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, label_pred),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "best_test_thresh": optimal_threshold
    }
    return metrics, y_true, y_pred, label_pred