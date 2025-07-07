import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple
from custom_lasso_regression import CustomLassoRegression
from helper_funcs import HelperFuncs
from regression_metrics import RegressionMetrics


def main():
    linear_data = pd.read_csv(
        "2. Building ML Models From Scratch/_datasets/linear_data.txt", header=None)
    X, y = np.array(linear_data.iloc[:, 0]), np.array(linear_data.iloc[:, 1])
    X = HelperFuncs.standardise(X)

    # Example grid of lambda values
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100]

    epochs = 10000
    alpha = 0.001
    # Run cross-validation
    cv_results = HelperFuncs.cross_validate_lasso(
        X, y, lambdas, k=5, alpha=0.001, epochs=10000, random_state=42)

    # Find the best lambda
    best_lambda = min(cv_results, key=cv_results.get)
    print('----------')
    print(f'Best lambda: {best_lambda}')
    print(f'Cross-validated MSEs: {cv_results}')

    model = CustomLassoRegression(
        w=0, b=0, alpha=alpha, epochs=epochs, lambda_=best_lambda)
    model.train(X, y)

    y_pred = model.predict(X)
    metrics = RegressionMetrics(y, y_pred)
    mse_custom, rmse_custom, mae_custom, r2_custom = metrics.evaluate()
    print(f'MSE (Custom): {mse_custom:.4f}')
    print(f'RMSE (Custom): {rmse_custom:.4f}')
    print(f'MAE (Custom): {mae_custom:.4f}')
    print(f'R-Squared (Custom): {r2_custom:.4f}')
    print('----------')


if __name__ == '__main__':
    main()
