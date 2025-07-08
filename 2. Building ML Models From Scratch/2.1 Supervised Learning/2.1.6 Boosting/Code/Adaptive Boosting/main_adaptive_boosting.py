import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from custom_adaptive_boosting import CustomAdaBoost


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    y = np.where(y == 0, -1, 1)  # AdaBoost expects labels as -1 and +1

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train AdaBoost
    model = CustomAdaBoost(n_weak_learners=10)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print(f"Test Accuracy (Custom): {accuracy:.4f}")


if __name__ == "__main__":
    main()
