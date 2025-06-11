import numpy as np
from typing import Callable
from typing import Tuple
from numpy.typing import NDArray

KernelFunctionType = Callable[[
    NDArray[np.float64], NDArray[np.float64]], float]


class CustomSVM:
    """
    A custom implementation of a Support Vector Machine (SVM) using Sequential Minimal Optimisation (SMO).

    Attributes:
        kernel_name (str): Name of the kernel function.
        kernel_func (KernelFunctionType): Kernel function.
        C (float): Regularisation parameter.
        tol (float): Tolerance for stopping criterion.
        max_passes (int): Maximum number of iterations without alpha updates.
        degree (int): Degree for polynomial kernel.
        coef0 (float): Independent term in polynomial and sigmoid kernels.
        gamma (float): Parameter for RBF, polynomial, and sigmoid kernels.
        sv_threshold (float): Threshold to identify support vectors.
        alpha_diff_threshold (float): Minimum change in alpha to be considered significant.
        alpha_sv (NDArray[np.float64]): Support vector Lagrange multipliers.
        X_sv (NDArray[np.float64]): Support vectors.
        y_sv (NDArray[np.int64]): Labels of the support vectors.
        b (float): Bias term.
        weighted_sv (NDArray[np.float64]): Precomputed product of alpha_sv and y_sv for efficiency.
    """

    def __init__(self, kernel: str = "linear", C: float = 1.0, tol: float = 1e-3,
                 max_passes: int = 10, degree: int = 3, coef0: float = 1.0,
                 gamma: float = 0.5, sv_threshold: float = 1e-8,
                 alpha_diff_threshold: float = 1e-8):
        """
        Initialise the SVM model with the given hyperparameters.

        Parameters:
            kernel (str): Kernel function to use ("linear", "poly", or "rbf").
            C (float): Regularisation parameter.
            tol (float): Tolerance for stopping criterion.
            max_passes (int): Maximum number of iterations without alpha updates.
            degree (int): Degree for polynomial kernel.
            coef0 (float): Independent term in polynomial kernel.
            gamma (float): Parameter for RBF kernel.
            sv_threshold (float): Threshold to identify support vectors.
            alpha_diff_threshold (float): Minimum change in alpha to be considered significant.
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.sv_threshold = sv_threshold
        self.alpha_diff_threshold = alpha_diff_threshold
        self.kernel_name = kernel
        self.kernel_func: KernelFunctionType = self.get_kernel_func(kernel)

        # Learned parameters
        self.alpha_sv = None
        self.X_sv = None
        self.y_sv = None
        self.b = 0
        self.weighted_sv = None  # Precomputed product for efficiency

    def get_kernel_func(self, kernel: str) -> KernelFunctionType:
        """
        Retrieve the kernel function based on the kernel name.

        Parameters:
            kernel (str): Name of the kernel.

        Returns:
            KernelFunctionType: Kernel function.
        """
        if kernel == "linear":
            return lambda x1, x2: np.dot(x1, x2)
        elif kernel == "poly":
            return lambda x1, x2: (np.dot(x1, x2) + self.coef0) ** self.degree
        elif kernel == "rbf":
            return lambda x1, x2: np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def compute_kernel_matrix(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Compute the kernel matrix for the dataset.

        Parameters:
            X (NDArray[np.float64]): Input dataset of shape (n_samples, n_features).

        Returns:
            NDArray[np.int64]: Kernel matrix.
        """
        return np.array([[self.kernel_func(x1, x2) for x2 in X] for x1 in X])

    def error(self, i: int, alphas: NDArray[np.float64], y: NDArray[np.int64],
              K: NDArray[np.int64], b: float) -> float:
        """
        Compute the error for the i-th sample.

        Parameters:
            i (int): Index of the sample.
            alphas (NDArray[np.float64]): Lagrange multipliers.
            y (NDArray[np.int64]): Labels.
            K (NDArray[np.int64]): Kernel matrix.
            b (float): Bias term.

        Returns:
            float: Error for the i-th sample.
        """
        return np.dot(alphas * y, K[:, i]) + b - y[i]

    def bounds(self, y_i: int, y_j: int, alpha_i: float, alpha_j: float) -> Tuple[float, float]:
        """
        Compute the bounds for alpha_j during optimisation.

        Parameters:
            y_i (int): Label for sample i.
            y_j (int): Label for sample j.
            alpha_i (float): Current alpha for sample i.
            alpha_j (float): Current alpha for sample j.

        Returns:
            Tuple[float, float]: Lower and upper bounds for alpha_j.
        """
        if y_i != y_j:
            return max(0, alpha_j - alpha_i), min(self.C, self.C + alpha_j - alpha_i)
        return max(0, alpha_i + alpha_j - self.C), min(self.C, alpha_i + alpha_j)

    def update_bias(self, b: float, Ei: float, Ej: float, y_i: int, y_j: int,
                    alpha_i: float, alpha_j: float, alpha_i_old: float,
                    alpha_j_old: float, K: NDArray[np.int64], i: int, j: int) -> float:
        """
        Update the bias term during the optimisation process.

        Parameters:
            b (float): Current bias.
            Ei (float): Error for sample i.
            Ej (float): Error for sample j.
            y_i (int): Label for sample i.
            y_j (int): Label for sample j.
            alpha_i (float): Updated alpha for sample i.
            alpha_j (float): Updated alpha for sample j.
            alpha_i_old (float): Previous alpha for sample i.
            alpha_j_old (float): Previous alpha for sample j.
            K (NDArray[np.int64]): Kernel matrix.
            i (int): Index of sample i.
            j (int): Index of sample j.

        Returns:
            float: Updated bias term.
        """
        b1 = b - Ei - y_i * (alpha_i - alpha_i_old) * \
            K[i, i] - y_j * (alpha_j - alpha_j_old) * K[i, j]
        b2 = b - Ej - y_i * (alpha_i - alpha_i_old) * \
            K[i, j] - y_j * (alpha_j - alpha_j_old) * K[j, j]
        if 0 < alpha_i < self.C:
            return b1
        elif 0 < alpha_j < self.C:
            return b2
        return (b1 + b2) / 2

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """
        Train the SVM model using Sequential Minimal Optimisation (SMO).

        Parameters:
            X (NDArray[np.float64]): Training dataset of shape (n_samples, n_features).
            y (NDArray[np.int64]): Labels of shape (n_samples,).
        """
        n = len(y)
        alphas = np.zeros(n)
        b = 0
        passes = 0

        # Kernel matrix (for pairwise kernel evaluations)
        K = self.compute_kernel_matrix(X)

        # Optimisation loop until convergence or max_passes is reached
        while passes < self.max_passes:
            # Track the number of alphas updated in this pass
            alpha_changed = 0

            # Iterate over each sample
            for i in range(n):
                # Compute the error for the current sample
                Ei = self.error(i, alphas, y, K, b)

                # Check if the sample violated the KKT conditions
                if (y[i] * Ei < -self.tol and alphas[i] < self.C) or (y[i] * Ei > self.tol and alphas[i] > 0):
                    # Randomly select a second sample (j) different from i
                    j = np.random.choice([x for x in range(n) if x != i])

                    # Compute the error for the second sample
                    Ej = self.error(j, alphas, y, K, b)

                    # Store the old values of alphas for i and j
                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                    # Compute the bounds L and H for the new value of alpha_j
                    L, H = self.bounds(y[i], y[j], alpha_i_old, alpha_j_old)
                    if L == H:
                        # If bounds are the same, skip this pair
                        continue

                    # Compute the second derivative (eta) of the objective function
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        # If eta is non-negative, skip this pair as it won't reduce the objective
                        continue

                    # Update alpha_j using the gradient of the objective function
                    alphas[j] -= y[j] * (Ei - Ej) / eta

                    # Clip alpha_j to ensure it lies within the bounds [L, H]
                    alphas[j] = np.clip(alphas[j], L, H)

                    # Check if the change in alpha_j is significant
                    if abs(alphas[j] - alpha_j_old) < self.alpha_diff_threshold:
                        # If the change is negligible, skip this pair
                        continue

                    # Update alpha_i based on the new value of alpha_j
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                    # Update the bias term to ensure the KKT conditions are satisfied
                    b = self.update_bias(b, Ei, Ej, y[i], y[j],
                                         alphas[i], alphas[j], alpha_i_old, alpha_j_old, K, i, j)

                    # +1 to the count of alpha updates
                    alpha_changed += 1

            # If no alphas were updated in this pass, increment passes
            passes += 1 if alpha_changed == 0 else 0

        # Identify support vectors (non-zero alphas)
        sv_mask = alphas > self.sv_threshold

        # Store support vector parameters
        self.alpha_sv = alphas[sv_mask]  # Non-zero Lagrange multipliers
        self.X_sv = X[sv_mask]           # Corresponding feature vectors
        self.y_sv = y[sv_mask]           # Corresponding labels
        self.b = b                       # Final bias term
        self._weighted_sv = self.alpha_sv * self.y_sv  # Precompute for decision function

    def decision_function(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the decision function for the input samples.

        Parameters:
            X_test (NDArray[np.float64]): Test samples of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Decision values for the test samples.
        """
        K = np.array([[self.kernel_func(x, sv)
                     for sv in self.X_sv] for x in X_test])
        return np.dot(K, self._weighted_sv) + self.b

    def predict(self, X_test: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Predict labels for the input samples.

        Parameters:
            X_test (NDArray[np.float64]): Test samples of shape (n_samples, n_features).

        Returns:
            NDArray[np.int64]: Predicted labels of shape (n_samples,).
        """
        return np.sign(self.decision_function(X_test))

    def accuracy(self, y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
        """
        Compute the accuracy of predictions.

        Parameters:
            y_true (NDArray[np.int64]): True labels.
            y_pred (NDArray[np.int64]): Predicted labels.

        Returns:
            float: Accuracy of the predictions.
        """
        return np.mean(y_pred == y_true)

    def get_margin_support_vectors(self, margin: float = 1.0, eps: float = 1e-3) -> NDArray[np.float64]:
        """
        Retrieve support vectors lying within the margin.

        Parameters:
            margin (float): Margin boundary (default is 1.0).
            eps (float): Tolerance for identifying support vectors.

        Returns:
            NDArray[np.float64]: Support vectors within the margin.
        """
        """Return support vectors lying within the margin zone."""
        decision_values = self.decision_function(self.X_sv)
        return self.X_sv[np.abs(np.abs(decision_values) - margin) < eps]
