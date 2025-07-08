import numpy as np
import pandas as pd
from custom_logistic_regression import CustomLogisticRegression


def main():
    data_logistic = pd.read_csv(
        "2. Building ML Models From Scratch/_datasets/logistic_data.txt", header=None
    )
    x_1 = np.array(data_logistic.iloc[:, 0])
    x_2 = np.array(data_logistic.iloc[:, 1])
    y = np.array(data_logistic.iloc[:, 2])
    # Combine features into X with an added bias column (first column)
    # X = np.c_[np.ones(x_1.shape[0]), x_1, x_2]
    X = np.concatenate(
        (np.ones((x_1.shape[0], 1)), x_1[:, np.newaxis], x_2[:, np.newaxis]), axis=1
    )
    y = y[:, np.newaxis]  # Reshape y for matrix operations

    # Initialise weights
    W = np.array([-10, 0.2, 0.2])[:, np.newaxis]

    # Initialise and train model
    model = CustomLogisticRegression(W=W, alpha=0.001, epochs=5000)
    model.train(X, y)

    # Plot loss function if needed
    # model.plot_loss_history()

    # Prediction with a test sample
    test_W = np.array([1, 72, 52])[np.newaxis, :]  # [bias, x_1, x_2]
    predicted_probability = model.predict_proba(test_W)
    predicted_label = model.predict(test_W)
    print(
        f"Predicted probability for the test sample: {predicted_probability[0, 0]:.5f}"
    )
    print(f"Predicted label for the test sample: {predicted_label[0, 0]}")


if __name__ == "__main__":
    main()
