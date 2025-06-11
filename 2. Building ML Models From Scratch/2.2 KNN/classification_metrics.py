import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


class ClassificationMetrics:
    """
    A class to compute and evaluate classification metrics (accuracy, precision, recall, and F1-score).
    """

    def __init__(self, y_true, y_pred):
        """
        Initialise the ClassificationMetrics instance.

        Args:
            y_true (NDArray[np.str_]): True labels of the dataset.
            y_pred (NDArray[np.str_]): Predicted labels of the dataset.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def confusion_matrix(self, class_names: List[str] = None) -> Tuple[NDArray[np.int64], List[str]]:
        """
        Calculate the confusion matrix.

        Args:
            class_names (List[str], optional): List of class names. Defaults to None.

        Returns:
            Tuple: 
            - NDArray[np.int64]: Confusion matrix.
            - List[str]: List of class names.
        """
        # Encode labels as integers
        unique_classes = np.unique(self.y_true)
        if class_names is None:
            class_names = [str(cls) for cls in unique_classes]
        class_to_index = {cls: i for i, cls in enumerate(unique_classes)}

        n_classes = len(unique_classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for true, pred in zip(self.y_true, self.y_pred):
            true_idx = class_to_index[true]
            pred_idx = class_to_index[pred]
            matrix[true_idx][pred_idx] += 1

        return matrix, class_names

    def accuracy(self):
        """
        Calculate the accuracy of predictions by comparing true and predicted labels.

        Returns:
            float: Classification accuracy as a percentage (0.0 to 100.0).
        """
        return np.mean(self.y_true == self.y_pred)

    def precision(self):
        """
        Calculate precision for each class.

        Returns:
            NDArray[np.float64]: Precision values for each class.
        """
        cm, _ = self.confusion_matrix()
        return np.diag(cm) / (np.sum(cm, axis=0) + 1e-7)

    def recall(self):
        """
        Calculate recall for each class.

        Returns:
            NDArray[np.float64]: Recall values for each class.
        """
        cm, _ = self.confusion_matrix()
        return np.diag(cm) / (np.sum(cm, axis=1) + 1e-7)

    def f1_score(self):
        """
        Calculate F1-score for each class.

        Returns:
            NDArray[np.float64]: F1-scores for each class.
        """
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec + 1e-7)

    def evaluate(self, class_names: List[str] = None) -> None:
        """
        Print evaluation metrics including accuracy, precision, recall, and F1-score for each class.

        Args:
            y_true (NDArray[np.str_]): True labels.
            y_pred (NDArray[np.str_]): Predicted labels.
            class_names (List[str], optional): List of class names. Defaults to None.
        """
        cm, class_names = self.confusion_matrix(class_names)
        acc = self.accuracy()
        prec = self.precision()
        rec = self.recall()
        f1 = self.f1_score()
        return acc, np.mean(prec), np.mean(rec), np.mean(f1), cm
