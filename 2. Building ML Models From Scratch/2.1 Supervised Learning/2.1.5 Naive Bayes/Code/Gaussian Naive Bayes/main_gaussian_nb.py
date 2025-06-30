import pandas as pd
from classification_metrics import ClassificationMetrics
from custom_gaussian_nb import CustomGaussianNB
from sklearn.datasets import load_iris
from helper_funcs import HelperFuncs


def main():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = CustomGaussianNB(epsilon=1e-9)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = ClassificationMetrics(y_test, y_pred)
    acc, prec, rec, f1, cm = metrics.evaluate()
    print(f'Accuracy (Test): {acc:.4f}')
    print(f'Precision (Test): {prec:.4f}')
    print(f'Recall (Test): {rec:.4f}')
    print(f'F1-Score (Test): {f1:.4f}')
    print(f'Confusion Matrix (Test):\n{cm}')


if __name__ == '__main__':
    main()
