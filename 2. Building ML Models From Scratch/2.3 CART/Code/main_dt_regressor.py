import pandas as pd
from regression_metrics import RegressionMetrics
from helper_funcs import HelperFuncs
from custom_dt_regressor import CustomDecisionTreeRegressor


def main():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/YBI-Foundation/Dataset/refs/heads/main/Admission%20Chance.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    feature_names = df.columns[:-1].tolist()  # All columns except the last one
    X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Train the decision tree
    tree = CustomDecisionTreeRegressor(max_depth=3, metric='variance')
    tree.fit(X_train, y_train, feature_names=feature_names)

    # Predict and evaluate
    y_pred = tree.predict(X_test)
    metrics = RegressionMetrics(y_test, y_pred)
    mse_custom, rmse_custom, mae_custom, r2_custom = metrics.evaluate()
    print(f'MSE (Custom): {mse_custom:.4f}')
    print(f'RMSE (Custom): {rmse_custom:.4f}')
    print(f'MAE (Custom): {mae_custom:.4f}')
    print(f'R-Squared (Custom): {r2_custom:.4f}')
    print('----------')

    # Single prediction
    sample = X_test.iloc[0]
    single_prediction = tree.predict(sample)
    print(
        f'Predicted: {single_prediction:.4f}, Actual: {y_test.iloc[0]:.4f}')
    print('----------')

    # Print the tree structure
    print('\nDecision Tree Structure:')
    tree.print_tree()


if __name__ == '__main__':
    main()
