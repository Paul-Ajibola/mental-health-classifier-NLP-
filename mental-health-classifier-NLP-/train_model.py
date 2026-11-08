"""Train and save the reproducible text-classification pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "mental_health.csv"
MODEL_PATH = BASE_DIR / "models" / "text_classifier.joblib"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    labels = set(data["label"].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels 0 and 1; found {sorted(labels)}.")

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=data["label"].astype(int),
    )

    classifier = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=2500, min_df=2, ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]
    )
    classifier.fit(x_train, y_train)

    print(classification_report(y_test, classifier.predict(x_test), digits=4))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    print(f"Saved model pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
