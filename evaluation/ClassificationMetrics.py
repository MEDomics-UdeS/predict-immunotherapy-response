import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


class ClassificationMetrics:
    """
    Implementation of some classification evaluation metrics.
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_roc_curve(y_true: np.ndarray[int],
                          y_score: np.ndarray[float]) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Compute the ROC curve associated to the prediction.

        ### Parameters :
        - y_true (n_samples, ) : numpy array containing the correct classes for each sample
        - y_score (n_samples, ): numpy array containing the output score predicted by the model (class 1 probability)

        ### Returns :
        - The False Positive Rate (x-axis) for each threshold
        - The True Positive Rate (y-axis) for each threshold
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        return fpr, tpr

    @staticmethod
    def compute_auc(y_true: np.ndarray[int],
                    y_score: np.ndarray[float]) -> float:
        """
        Compute the AUC associated to the prediction.

        ### Parameters :
        - y_true (n_samples, ) : numpy array containing the correct classes for each sample
        - y_score (n_samples, ): numpy array containing the output score predicted by the model (class 1 probability)

        ### Returns :
        The AUC score
        """
        return np.round(roc_auc_score(y_true, y_score), 2)

    @staticmethod
    def compute_sensitivity_specificity(y_true: np.ndarray[int],
                                        y_pred: np.ndarray[int]) -> tuple[float, float]:
        """
        Compute the sensitivity and the specificity of the prediction.

        ### Parameters :
        - y_true (n_samples, ) : numpy array containing the correct class for each sample
        - y_pred (n_samples, ): numpy array containing the predicted class for each sample

        ### Returns :
        - The sensitivity
        - The spepcificity
        """
        # Compute the confusion matrix
        TP, FP, FN, TN = confusion_matrix(y_true, y_pred).ravel()

        # Compute sensitivity
        sensitivity = np.round(TP / (TP + FN), 2)

        # Compute specificity
        specificity = np.round(TN / (TN + FP), 2)

        return sensitivity, specificity
