import pandas as pd
from custom_categorical_nb import CustomCategoricalNB
from classification_metrics import ClassificationMetrics


def main():
    df = pd.read_csv(
        '2. Building ML Models From Scratch/_datasets/material.csv', index_col=0)
    X, y = df.drop('demand', axis=1), df['demand']

    model = CustomCategoricalNB()
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = ClassificationMetrics(y_pred, y)
    acc, prec, rec, f1, cm = metrics.evaluate()
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{cm}')


if __name__ == '__main__':
    main()
