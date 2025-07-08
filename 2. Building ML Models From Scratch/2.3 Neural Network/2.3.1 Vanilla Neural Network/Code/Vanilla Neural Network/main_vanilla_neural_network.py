import numpy as np
from vanilla_neural_network import VanillaNeuralNetwork


def main():
    # XOR dataset (inputs and outputs)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialise and train network
    nn = VanillaNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    nn.train(X, y, epochs=10000, lr=0.1)

    # Test predictions
    print("\nFinal Predictions:")
    for i in range(len(X)):
        probability = round(nn.forward(X[i : i + 1]).item(), 4)
        output = 0 if probability <= 0.5 else 1
        print(
            f"Input: {X[i]} -> Output: {output} (Probability: {probability}) | Expected: {y[i][0]}"
        )


if __name__ == "__main__":
    main()
