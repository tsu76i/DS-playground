import pandas as pd
from sklearn.model_selection import train_test_split
from custom_extreme_gradient_boosting import CustomXGBoost
from regression_metrics import RegressionMetrics


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/YBI-Foundation/Dataset/refs/heads/main/Admission%20Chance.csv"
    )
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train AdaBoost
    model = CustomXGBoost(
        n_estimators=200, learning_rate=0.1, max_depth=2, min_samples_leaf=1
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = RegressionMetrics(y_test, y_pred)
    mse, rmse, mae, r2 = metrics.evaluate()
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R-Squared: {r2:.4f}")
    print("----------")


if __name__ == "__main__":
    main()
