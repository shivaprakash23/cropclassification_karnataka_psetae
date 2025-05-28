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
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed



def confusion_matrix_analysis(mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    Args:
        mat (array): confusion matrix

    Returns:
        per_class (dict) : per class metrics
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

        # Handle division by zero cases
        denominator = tp + fp + fn
        d['IoU'] = tp / denominator if denominator > 0 else 0.0
        
        denominator = tp + fp
        d['Precision'] = tp / denominator if denominator > 0 else 0.0
        
        denominator = tp + fn
        d['Recall'] = tp / denominator if denominator > 0 else 0.0
        
        denominator = 2 * tp + fp + fn
        d['F1-score'] = 2 * tp / denominator if denominator > 0 else 0.0

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    # Handle division by zero cases for micro-averaged metrics
    denominator = TP + FP + FN
    overall['micro_IoU'] = TP / denominator if denominator > 0 else 0.0
    
    denominator = TP + FP
    overall['micro_Precision'] = TP / denominator if denominator > 0 else 0.0
    
    denominator = TP + FN
    overall['micro_Recall'] = TP / denominator if denominator > 0 else 0.0
    
    denominator = 2 * TP + FP + FN
    overall['micro_F1-score'] = 2 * TP / denominator if denominator > 0 else 0.0

    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = macro.loc['IoU']
    overall['MACRO_Precision'] = macro.loc['Precision']
    overall['MACRO_Recall'] = macro.loc['Recall']
    overall['MACRO_F1-score'] = macro.loc['F1-score']

    overall['Accuracy'] = np.sum(np.diag(mat)) / np.sum(mat)

    return per_class, overall
