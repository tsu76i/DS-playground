import pandas as pd
from tqdm import tqdm
from helper_funcs import HelperFuncs
from regression_metrics import RegressionMetrics
from custom_knn_regressor import CustomKNNRegressor


def main():
    # Import data
    df = pd.read_csv(
        '2. Building ML Models From Scratch/_datasets/diamonds.csv')

    # Data Pre-processing
    df = HelperFuncs.remove_outliers_iqr(df)
    categorical_features = {
        'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
        'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
        'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    }
    for feature, levels in categorical_features.items():
        df[feature] = HelperFuncs.ordinal_encode(df[feature], levels)

    continuous_features = ['carat', 'x', 'y', 'z',
                           'depth', 'table', 'cut', 'color', 'clarity']
    df[continuous_features] = HelperFuncs.standardise(
        df[continuous_features])
    df[continuous_features] = HelperFuncs.normalise(
        df[continuous_features])

    # Prepare data
    MSE_list_custom, RMSE_list_custom, MAE_list_custom, R2_list_custom = [], [], [], []
    X = df.drop(columns=['price'])
    y = df['price']
    k_range = range(1, 21)  # Generate k-values from 1 to 20

    # Split data
    X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train KNN regressor, make predictions and evaluate the model for different k values.
    for k in tqdm(k_range):
        knn_reg = CustomKNNRegressor(k=k)
        knn_reg.fit(X_train, y_train)
        y_pred = knn_reg.predict(X_test)

        metrics = RegressionMetrics(y_test, y_pred)
        mse, rmse, mae, r2 = metrics.evaluate()
        print(f"For k = {k}, MSE: {mse}")
        print(f"For k = {k}, RMSE: {rmse}")
        print(f"For k = {k}, MAE: {mae}")
        print(f"For k = {k}, R2: {r2}")
        print("-----------")

    # Prediction of a sample
    x_single_2d = X_test.iloc[0]
    y_pred = knn_reg.predict(x_single_2d.values)
    y_pred_scalar = y_pred[0]
    print("Predicted value:", y_pred_scalar)


if __name__ == '__main__':
    main()
