import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class HelperFuncs:
    def standardise_data(X: NDArray[np.float64]) -> NDArray[np.float64]:
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def plot_explained_variance(ev_ratio, cev):
        # Plot the graphs
        plt.figure(figsize=(12, 6))

        # Percentage of explained variance
        plt.subplot(1, 2, 1)
        plt.bar(
            range(1, len(ev_ratio) + 1),
            ev_ratio,
            alpha=0.7,
            color="b",
            label="Explained Variance",
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Ratio")
        plt.title("Explained Variance by Principal Component")
        plt.grid(True)

        # Cumulative explained variance
        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(cev) + 1),
            cev,
            marker="o",
            color="r",
            label="Cumulative Variance",
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Cumulative Variance Ratio")
        plt.title("Cumulative Explained Variance")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
