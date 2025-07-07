import numpy as np
import pandas as pd
from custom_categorical_gradient_boosting import CustomCatBoost
from classification_metrics import ClassificationMetrics


def main():
    # Import data
    df = pd.read_csv(
        '2. Building ML Models From Scratch/_datasets/material.csv')
    X = df.drop('demand', axis=1)
    y = df['demand']

    cat_features = ['size', 'material', 'color', 'sleeves']
    target = 'demand'

    # Encode target as integer
    target_map = {v: i for i, v in enumerate(df[target].unique())}
    df['target_enc'] = df[target].map(target_map)

    # Instantiate model
    model_custom = CustomCatBoost(
        n_estimators=20,
        learning_rate=0.2,
        max_depth=4,
        min_samples_split=5
    )

    # Apply ordered target encoding
    df_enc = model_custom.apply_ordered_target_encoding(
        df, cat_features, 'target_enc')

    # Prepare data
    X = df_enc[cat_features].values.astype(np.float64)
    y = df['target_enc'].values.astype(np.int64)
    n_classes = len(target_map)

    # Fit model
    model_custom.fit(X, y, n_classes, cat_cols=None, df=None, target_col=None)

    # Predict and evaluate
    y_pred, y_proba = model_custom.predict(X, n_classes)
    metrics = ClassificationMetrics(y, y_pred)
    acc_custom, prec_custom, rec_custom, f1_custom, cm_custom = metrics.evaluate()
    print(f'Accuracy: (Custom) {acc_custom:.4f}')
    print(f'Precision: (Custom) {prec_custom:.4f}')
    print(f'Recall (Custom): {rec_custom:.4f}')
    print(f'F1-Score (Custom): {f1_custom:.4f}')
    print(f'Confusion Matrix (Custom):\n{cm_custom}')


if __name__ == '__main__':
    main()
