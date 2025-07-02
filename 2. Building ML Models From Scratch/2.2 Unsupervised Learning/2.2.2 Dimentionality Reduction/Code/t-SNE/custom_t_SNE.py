import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from typing import Tuple
import random


class CustomTSNE:
    """
    Custom implementation of t-Distributed Stochastic Neighbour Embedding (t-SNE).

    Attributes:
        embedding_: Optional[NDArray[np.float64]]
            Store the final embedding after calling fit_transform (shape: [n_samples, dim])
    """

    def __init__(self, dim=2, perplexity=30.0, lr=200, max_iter=1000,
                 early_exaggeration=4.0, exaggeration_iter=100,
                 momentum=0.5, final_momentum=0.8, random_state=None) -> None:
        """
        Initialise t-SNE parameters.

        Args:
            dim: Output dimensionality of the embedding (typically 2 or 3). Default=2.
            perplexity: Target perplexity for probability distributions in high-dimensional space. Default=30.0.
            lr: Learning rate for gradient descent optimisation. Default=200.
            max_iter: Number of optimisation iterations. Default=1000.
            early_exaggeration:
                Scaling factor applied to P-matrix to prevent tight clusters and to escape local minima. Default=4.0.
            exaggeration_iter: Iteration at which early exaggeration stops and P-matrix is scaled down. Default=100.
            momentum: Momentum factor for gradient updates. Accelerates optimisation in relevant directions. Default=0.5
            final_momentum: Momentum factor for gradient updates after early exaggeration phase. Default=0.8.
            random_state: Seed for random number generator to ensure reproducible results.
        """
        self.dim = dim
        self.perplexity = perplexity
        self.lr = lr
        self.max_iter = max_iter
        self.early_exaggeration = early_exaggeration
        self.exaggeration_iter = exaggeration_iter
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.random_state = random_state
        self.embedding_ = None

    def _shannon_and_p_dist(self, D_i: NDArray[np.float64], beta: float) -> Tuple[float, NDArray[np.float64]]:
        """
        Compute Shannon entropy and probability distribution with numerical stability.

        Args:
            D_i: Squared Euclidean distances from point i to others (shape: [n-1]).
            beta: Precision parameter (inverse variance).

        Returns:
            H: Shannon entropy in bits.
            P: Probability distribution (shape: [n-1]).
        """
        P = np.exp(-D_i * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            P = np.zeros_like(P)
            H = 0
        else:
            P /= P_sum
            H = -np.sum(P * np.log2(P + 1e-10))
        return H, P

    def _binary_search_beta(self, D_i: NDArray[np.float64], target_entropy: float,
                            tol: float = 1e-5, max_iter: int = 50,
                            initial_beta: float = 1.0) -> NDArray[np.float64]:
        """
        Binary search for beta that produces target entropy.

        Args:
            D_i: Squared Euclidean distances (shape: [n-1]).
            target_entropy: Target Shannon entropy (log2(perplexity)).
            tol: Tolerance for convergence.
            max_iter: Maximum binary search iterations.
            initial_beta: Starting beta value.

        Returns:
            Probability distribution for the point (shape: [n-1]).
        """
        beta = initial_beta
        beta_min, beta_max = -np.inf, np.inf
        H, proba_distribution = self._shannon_and_p_dist(D_i, beta)
        H_diff = H - target_entropy
        i = 0

        while np.abs(H_diff) > tol and i < max_iter:
            if H_diff > 0:
                beta_min = beta
                if np.isinf(beta_max):
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if np.isinf(beta_min):
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

            H, proba_distribution = self._shannon_and_p_dist(D_i, beta)
            H_diff = H - target_entropy
            i += 1

        return proba_distribution

    def _pairwise_affinities(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute high-dimensional affinity matrix P.

        Args:
            X: Input data matrix (shape: [n_samples, n_features]).

        Returns:
            P: Symmetrised affinity matrix (shape: [n_samples, n_samples]).
        """
        n = X.shape[0]
        distances = squareform(pdist(X, "sqeuclidean"))
        P = np.zeros((n, n))
        target_entropy = np.log2(self.perplexity)

        for i in range(n):
            D_i = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            proba_dist = self._binary_search_beta(D_i, target_entropy)
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = proba_dist

        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        return P

    def _low_dimensional_affinities(self, Y: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute low-dimensional affinity matrix Q and distance matrix D.

        Args:
            Y: Low-dimensional embeddings (shape: [n_samples, dim]).

        Returns:
            Q: Low-dimensional affinity matrix (shape: [n_samples, n_samples]).
            D: Squared Euclidean distance matrix (shape: [n_samples, n_samples]).
        """
        sum_Y = np.sum(Y**2, axis=1)
        D = sum_Y[:, None] + sum_Y[None, :] - 2 * np.dot(Y, Y.T)
        Q = 1 / (1 + D)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        return Q, D

    def _gradient(self, P: NDArray[np.float64], Q: NDArray[np.float64],
                  Y: NDArray[np.float64], D: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute gradient of KL divergence.

        Args:
            P: High-dimensional affinity matrix.
            Q: Low-dimensional affinity matrix.
            Y: Current embeddings.
            D: Squared Euclidean distance matrix.

        Returns:
            Gradient matrix (shape: [n_samples, dim]).
        """
        PQ = P - Q
        inv_distances = 1 / (1 + D)
        np.fill_diagonal(inv_distances, 0)
        weighted_differences = np.expand_dims(
            PQ * inv_distances, axis=2) * (Y[:, None, :] - Y[None, :, :])
        return 4 * np.sum(weighted_differences, axis=1)

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute t-SNE embedding of input data.

        Args:
            X: Input data matrix (shape: [n_samples, n_features]).

        Returns:
            Y: Low-dimensional embedding (shape: [n_samples, dim]).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        n = X.shape[0]
        Y = np.random.normal(0, 1e-4, (n, self.dim))
        P = self._pairwise_affinities(X)
        P *= self.early_exaggeration
        velocity = np.zeros_like(Y)

        for i in range(self.max_iter):
            Q, D = self._low_dimensional_affinities(Y)
            grad = self._gradient(P, Q, Y, D)

            if i == self.exaggeration_iter:
                P /= self.early_exaggeration

            mom = self.final_momentum if i > self.exaggeration_iter else self.momentum
            velocity = mom * velocity - self.lr * grad
            Y += velocity

            if i % 100 == 0 or i == self.max_iter - 1:
                loss = np.sum(P * np.log((P + 1e-12) / (Q + 1e-12)))
                print(f"Iteration {i}: KL divergence = {loss:.4f}")

        self.embedding_ = Y
        return Y
