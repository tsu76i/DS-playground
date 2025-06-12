import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from numpy.typing import NDArray
from helper_funcs import HelperFuncs
from custom_pca import CustomPCA


def main():
    # Load data
    wine_data = load_wine(as_frame=True)
    X = wine_data.data

    # Standardise X
    X_standardised = HelperFuncs.standardise_data(X)

    # Fit
    custom_pca = CustomPCA(n_components=X.shape[1])
    custom_pca.fit(X_standardised)

    # Plot EV & CEV
    explained_variance_ratio = custom_pca.explained_variance_ratio
    cumulative_explained_variance = custom_pca.cumulative_explained_variance

    HelperFuncs.plot_explained_variance(explained_variance_ratio,
                                        cumulative_explained_variance)

    # Re-initialise and fit with n_components = 8
    custom_pca = CustomPCA(n_components=8)
    custom_pca.fit(X_standardised)
    X_transformed_custom = custom_pca.transform(X_standardised)

    # Print transformed data of 8 dimensions
    print(X_transformed_custom)


if __name__ == '__main__':
    main()
