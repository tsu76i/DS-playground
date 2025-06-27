from sklearn.datasets import load_breast_cancer
import pandas as pd
from helper_funcs import HelperFuncs
from classification_metrics import ClassificationMetrics
from custom_rf_classifier import CustomRandomForestClassifier


def main():
    # Load dataset
    data = load_breast_cancer()
    feature_names = data.feature_names.tolist()
    class_names = data.target_names.tolist()
    X, y = data.data, data.target
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = y
    X, y = df.drop('diagnosis', axis=1), df['diagnosis']

    # Split dataset
    X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Train the decision tree
    tree = CustomRandomForestClassifier(max_depth=3, metric='gini')
    tree.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = tree.predict(X_test)
    metrics = ClassificationMetrics(y_test, y_pred)
    acc_custom, prec_custom, rec_custom, f1_custom, cm_custom = metrics.evaluate()
    print(f'Accuracy (Custom): {acc_custom:.4f}')
    print(f'Precision: (Custom) {prec_custom:.4f}')
    print(f'Recall (Custom): {rec_custom:.4f}')
    print(f'F1-Score (Custom): {f1_custom:.4f}')
    print(f'Confusion Matrix (Custom):\n{cm_custom}')
    print('--------------------')


if __name__ == '__main__':
    main()
