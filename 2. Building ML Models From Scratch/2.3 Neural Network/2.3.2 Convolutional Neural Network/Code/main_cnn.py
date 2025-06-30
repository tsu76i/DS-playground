from conv_net import ConvNet
import numpy as np
from helper_funcs import HelperFuncs
from sklearn.datasets import fetch_openml


def main():

    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images, labels = mnist.data.reshape(-1,
                                        28, 28), mnist.target.astype(np.uint8)

    # Standard train/test split = 80%:20%
    split_idx = int(len(images)*0.8)
    X_train, y_train = images[:split_idx], labels[:split_idx]
    X_test, y_test = images[split_idx:], labels[split_idx:]

    X_train, y_train = HelperFuncs.preprocess_data(X_train, y_train)
    X_test, y_test = HelperFuncs.preprocess_data(X_test, y_test)

    # Training Setup
    epochs = 3
    batch_size = 64
    learning_rate = 0.01

    # Initialise model
    model = ConvNet()

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            logits = model.forward(X_batch)
            y_pred = HelperFuncs.softmax(logits)
            loss = HelperFuncs.cross_entropy_loss(y_pred, y_batch)

            # Backward pass
            grad = (y_pred - y_batch) / batch_size
            model.backward(grad)
            model.update_params(learning_rate)

            # Print metrics
            if i % 6400 == 0:
                acc = HelperFuncs.accuracy(y_pred, y_batch)
                print(
                    f"Epoch {epoch+1}, Batch {i//batch_size}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    # Evaluation
    test_pred = HelperFuncs.softmax(model.forward(X_test))
    test_acc = HelperFuncs.accuracy(test_pred, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
