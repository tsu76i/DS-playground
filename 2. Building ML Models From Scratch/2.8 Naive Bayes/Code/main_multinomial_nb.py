import pandas as pd
from custom_multinomial_nb import CustomMultinomialNB
from classification_metrics import ClassificationMetrics
from helper_funcs import HelperFuncs
from sklearn.feature_extraction.text import CountVectorizer


def main():
    df = pd.read_csv(
        '2. Building ML Models From Scratch/_datasets/spam.csv')
    df['clean_text'] = df['Message'].apply(
        HelperFuncs.clean_text)  # Clean data
    X, y = df['clean_text'], df['Category']

    # Transform X in Bag of Words
    vectoriser = CountVectorizer()
    X_doc_term = vectoriser.fit_transform(X)
    X = pd.DataFrame(X_doc_term.toarray(),
                     columns=vectoriser.get_feature_names_out())

    X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = CustomMultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = ClassificationMetrics(y_test, y_pred)
    acc, prec, rec, f1, cm = metrics.evaluate()
    print(f'Accuracy (Test): {acc:.4f}')
    print(f'Precision (Test): {prec:.4f}')
    print(f'Recall (Test): {rec:.4f}')
    print(f'F1-Score (Test): {f1:.4f}')
    print(f'Confusion Matrix (Test):\n{cm}')


if __name__ == '__main__':
    main()
