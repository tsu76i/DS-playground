import numpy as np
import pandas as pd
from typing import Tuple, List
from numpy.typing import NDArray


class ClassificationMetrics:
    """
    A class to compute and evaluate classification metrics (accuracy, precision, recall, and F1-score).
    """

    def __init__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.int8]) -> None:
        """
        Initialise the ClassificationMetrics instance.

        Args:
            y_true: True labels of the dataset.
            y_pred: Predicted labels of the dataset.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def confusion_matrix(self, class_names: List[str] = None) -> Tuple[NDArray[np.int64], List[str]]:
        """
        Calculate the confusion matrix.

        Args:
            class_names: List of class names. Defaults to None.

        Returns:
            - Confusion matrix.
            - List of class names.
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

    def accuracy(self) -> NDArray[np.float64]:
        """
        Calculate the accuracy of predictions by comparing true and predicted labels.

        Returns:
            Classification accuracy (0.0 to 1.0).
        """
        return np.mean(self.y_true == self.y_pred)

    def precision(self) -> NDArray[np.float64]:
        """
        Calculate precision for each class.

        Returns:
            Precision values for each class.
        """
        cm, _ = self.confusion_matrix()
        return np.diag(cm) / (np.sum(cm, axis=0) + 1e-7)

    def recall(self) -> NDArray[np.float64]:
        """
        Calculate recall for each class.

        Returns:
            Recall values for each class.
        """
        cm, _ = self.confusion_matrix()
        return np.diag(cm) / (np.sum(cm, axis=1) + 1e-7)

    def f1_score(self) -> NDArray[np.float64]:
        """
        Calculate F1-score for each class.

        Returns:
            F1-scores for each class.
        """
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec + 1e-7)

    def evaluate(self, class_names: List[str] = None) -> Tuple[np.float64, np.float64, np.float64, np.float64, NDArray[np.int64]]:
        """
        Calculate evaluation metrics including accuracy, precision, recall, and F1-score for each class.

        Args:
            class_names: List of class names. Defaults to None.

        Returns:
            Tuple:
            - Overall accuracy.
            - Average precision.
            - Average recall.
            - Average F1-score.
            - Confusion matrix.
        """
        cm, class_names = self.confusion_matrix(class_names)
        acc = self.accuracy()
        prec = self.precision()
        rec = self.recall()
        f1 = self.f1_score()
        return acc, np.mean(prec), np.mean(rec), np.mean(f1), cm
