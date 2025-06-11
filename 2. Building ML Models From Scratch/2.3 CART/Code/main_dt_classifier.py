from sklearn.datasets import load_breast_cancer
from helper_funcs import HelperFuncs
from classification_metrics import ClassificationMetrics
from custom_dt_classifier import CustomDecisionTreeClassifier


def main():
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names

    # Split dataset
    X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Train the decision tree
    tree = CustomDecisionTreeClassifier(max_depth=3, metric="gini")
    tree.fit(X_train, y_train, feature_names=feature_names,
             class_names=class_names)

    # Predict and evaluate
    y_pred = tree.predict(X_test)

    # Single prediction
    sample = X_test[0]
    single_prediction = tree.predict(sample)
    print(
        f"Predicted: {class_names[single_prediction]}, Actual: {class_names[y_test[0]]}")
    metrics = ClassificationMetrics(y_test, y_pred)
    acc_custom, prec_custom, rec_custom, f1_custom, cm_custom = metrics.evaluate()
    print(f"Accuracy (Custom): {acc_custom:.4f}")
    print(f"Precision: (Custom) {prec_custom:.4f}")
    print(f"Recall (Custom): {rec_custom:.4f}")
    print(f"F1-Score (Custom): {f1_custom:.4f}")
    print(f"Confusion Matrix (Custom):\n{cm_custom}")
    print('--------------------')

    # Print the tree structure
    print("\nDecision Tree Structure:")
    tree.print_tree()


if __name__ == '__main__':
    main()
