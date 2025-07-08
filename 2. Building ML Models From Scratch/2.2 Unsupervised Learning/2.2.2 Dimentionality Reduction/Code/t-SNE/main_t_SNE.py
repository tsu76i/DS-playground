from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE
import time
from custom_t_SNE import CustomTSNE
from helper_funcs import HelperFuncs


def main():
    # Generate Swiss Roll data
    data, colour = make_swiss_roll(n_samples=1000, noise=0.05)

    # Custom t-SNE
    print("Running custom t-SNE...")
    start = time.time()
    model = CustomTSNE(dim=2, max_iter=1000, perplexity=30, random_state=0, lr=200)
    custom_tsne = model.fit_transform(data)
    custom_time = time.time() - start

    # Scikit-learn t-SNE
    print("\nRunning scikit-learn t-SNE...")
    start = time.time()
    sk_tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        learning_rate=200,
        random_state=0,
        method="exact",
        init="random",
        verbose=1,
    ).fit_transform(data)
    sk_time = time.time() - start

    print(f"\nCustom t-SNE time: {custom_time:.2f}s")
    print(f"Scikit-learn t-SNE time: {sk_time:.2f}s")

    HelperFuncs.plot_tsne_comparison(custom_tsne, sk_tsne, colour)


if __name__ == "__main__":
    main()
