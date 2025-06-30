from custom_knn_classifier import CustomKNNClassifier
from classification_metrics import ClassificationMetrics
import pandas as pd
from helper_funcs import HelperFuncs


def main():
    df = pd.read_csv('2. Building ML Models From Scratch/_datasets/iris.csv')
    k = 7
    for feature in ['sepal', 'petal']:
        # Separate features and labels
        X = df[[f'{feature}_length', f'{feature}_width']]
        y = df['species']

        # Split the dataset
        X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Initialise and fit train data to KNN Classifier
        knn_classifier = CustomKNNClassifier(k=k)
        knn_classifier.fit(X_train, y_train)

        # Make predictions and calculate accuracy of the model
        y_pred = knn_classifier.predict(X_test)
        metrics = ClassificationMetrics(y_test, y_pred)
        acc, prec, rec, f1, cm = metrics.evaluate()
        print(f"Metrics for '{feature}' features with k = {k}:")
        print(f"Accuracy (Custom): {acc:.4f}")
        print(f"Precision: (Custom) {prec:.4f}")
        print(f"Recall (Custom): {rec:.4f}")
        print(f"F1-Score (Custom): {f1:.4f}")
        print(f"Confusion Matrix (Custom):\n{cm}")
        print('--------------------')


if __name__ == '__main__':
    main()
