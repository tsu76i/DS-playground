import pandas as pd
import numpy as np
from custom_bernoulli_nb import CustomBernoulliNB
from classification_metrics import ClassificationMetrics


def main():
    df = pd.read_csv(
        '2. Building ML Models From Scratch/_datasets/weather_forecast.csv')
    X, y = df.drop('Play', axis=1), df['Play']
    X_binary = pd.get_dummies(X)
    y_binary = pd.Series(np.where(y == 'Yes', 1, 0), name='Play')

    model = CustomBernoulliNB()
    model.fit(X_binary, y_binary)

    y_pred = model.predict(X_binary)
    metrics = ClassificationMetrics(y_pred, y_binary)
    acc, prec, rec, f1, cm = metrics.evaluate()
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{cm}')


if __name__ == '__main__':
    main()
