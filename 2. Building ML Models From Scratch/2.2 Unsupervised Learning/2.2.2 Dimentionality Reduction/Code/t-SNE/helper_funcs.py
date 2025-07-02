import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class HelperFuncs:
    def plot_tsne_comparison(custom_tsne: NDArray[np.float64], sk_tsne: NDArray[np.float64], colour: NDArray[np.float64]) -> None:
        """
        Visualises and compares two t-SNE embeddings side by side with colourbars.

        Args:
            custom_tsne: Low-dimensional embedding from custom t-SNE (shape: [n_samples, 2]).
            sk_tsne: Low-dimensional embedding from scikit-learn t-SNE (shape: [n_samples, 2]).
            colour: Unrolled positions of each point along the Swiss roll. (shape: [n_samples,])
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sc1 = ax1.scatter(custom_tsne[:, 0], custom_tsne[:,
                                                         1], c=colour, cmap='tab10', s=5)
        ax1.set_title('Custom t-SNE Implementation')
        fig.colorbar(sc1, ax=ax1)

        sc2 = ax2.scatter(sk_tsne[:, 0], sk_tsne[:, 1],
                          c=colour, cmap='tab10', s=5)
        ax2.set_title('Scikit-learn t-SNE')
        fig.colorbar(sc2, ax=ax2)

        plt.tight_layout()
        plt.show()
