import numpy as np
import pandas as pd
from custom_linear_regression import CustomLinearRegression


def main():
    # Loading data
    linear_data = pd.read_csv(
        "2. Building ML Models From Scratch/_datasets/linear_data.txt", header=None
    )

    # Create arrays as X (features) and y (target)
    X, y = np.array(linear_data.iloc[:, 0]), np.array(linear_data.iloc[:, 1])

    # Initialise and train model
    model = CustomLinearRegression(w=5, b=3, alpha=0.01, epochs=5000)
    model.train(X, y)

    # Plot loss function if needed
    # model.plot_loss_history()

    # Single sample prediction
    test_x = 15
    predicted_value = model.predict(test_x)
    print(f"Predicted value for X = {test_x}: {predicted_value:.5f}")
    model.plot_prediction(X, y, test_x, predicted_value)


if __name__ == "__main__":
    main()
