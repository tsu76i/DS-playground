from sklearn.datasets import load_wine
from helper_funcs import HelperFuncs
from custom_pca import CustomPCA


def main():
    wine_data = load_wine(as_frame=True)  # Load data
    X = wine_data.data

    X_standardised = HelperFuncs.standardise_data(X)  # Standardise X

    custom_pca = CustomPCA(n_components=X.shape[1])  # Initialise CustomPCA
    custom_pca.fit(X_standardised)  # Fit

    explained_variance_ratio = custom_pca.explained_variance_ratio
    cumulative_explained_variance = custom_pca.cumulative_explained_variance

    HelperFuncs.plot_explained_variance(explained_variance_ratio,
                                        cumulative_explained_variance)  # Plot EV & CEV

    # Re-initialise and fit with n_components = 8
    custom_pca = CustomPCA(n_components=8)
    custom_pca.fit(X_standardised)
    X_transformed_custom = custom_pca.transform(X_standardised)

    # Print transformed data of 8 dimensions
    print(X_transformed_custom)


if __name__ == '__main__':
    main()
