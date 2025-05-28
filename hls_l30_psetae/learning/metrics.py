"""
Metrics for evaluating HLS L30 model performance
"""

import numpy as np
import pandas as pd


def mIou(y_true, y_pred, n_classes):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average.
    
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
        
    Returns:
        mean Iou (float)
    """
    # Convert inputs to numpy arrays if they aren't already
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
        
    # Ensure integer type
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (y_true == i).astype(np.int32)
        y_p = (y_pred == i).astype(np.int32)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(np.int32))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed if n_observed > 0 else 0.0


def confusion_matrix_analysis(mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    
    Args:
        mat (array): confusion matrix

    Returns:
        per_class (dict): per class metrics
        overall (dict): overall metrics
    """
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = np.sum(mat[j, j])
        fp = np.sum(mat[:, j]) - tp
        fn = np.sum(mat[j, :]) - tp

        d['IoU'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        d['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        d['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        d['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall['micro_IoU'] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    overall['micro_Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
    overall['micro_Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0
    overall['micro_F1-score'] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = macro.loc['IoU']
    overall['MACRO_Precision'] = macro.loc['Precision']
    overall['MACRO_Recall'] = macro.loc['Recall']
    overall['MACRO_F1-score'] = macro.loc['F1-score']

    overall['Accuracy'] = np.sum(np.diag(mat)) / np.sum(mat) if np.sum(mat) > 0 else 0

    return per_class, overall
