import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from helper_funcs import HelperFuncs
from custom_svm import CustomSVM


def main():
    # Load datasets
    # Linear dataset (make_blobs)
    X_linear, y_linear = make_blobs(
        n_samples=200, centers=2, random_state=42, cluster_std=1.8)
    y_linear = np.where(y_linear == 0, -1, 1)  # Convert labels to -1, 1

    # Non-linear dataset (make_moons)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    y_moons = np.where(y_moons == 0, -1, 1)

    # Non-linear dataset (make_circles)
    X_circles, y_circles = make_circles(
        n_samples=200, factor=0.3, noise=0.1, random_state=42)
    y_circles = np.where(y_circles == 0, -1, 1)

    # Existing datasets
    datasets = {
        "Blobs": (X_linear, y_linear),
        "Moons": (X_moons, y_moons),
        "Circles": (X_circles, y_circles)
    }

    # Kernel settings
    kernel_settings = {
        "linear": {},
        "poly": {"degree": 3, "coef0": 1},
        "rbf": {"gamma": 1.0}
    }

    for name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
            X, y, test_size=0.2, random_state=42)
        print(f"\n=== Dataset: {name} ===")
        for kernel_name, params in kernel_settings.items():
            model = CustomSVM(kernel=kernel_name, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = model.accuracy(y_test, y_pred)
            print(
                f"{kernel_name.capitalize()} Kernel -> Test Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    main()
