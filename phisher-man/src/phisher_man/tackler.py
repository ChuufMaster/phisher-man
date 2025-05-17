from phisher_man.util import (
    clean_text,
    contains_url,
    count_urls,
    similar,
    remove_stopwords,
)
from phisher_man.evaluator import evaluate_model
import glob
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

import joblib


def load_and_normalize_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"Reading {file_path}")

    # Normalize mandatory columns
    if (
        "subject" not in df.columns
        or "body" not in df.columns
        or "label" not in df.columns
    ):
        raise ValueError(f"{file_path} is missing mandatory columns.")

    df = df.rename(columns=str.lower)

    # Ensure required columns exist
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["label"] = df["label"].astype(int)

    # Optional columns - default to None or 0
    df["urls"] = df["urls"] if "urls" in df.columns else 0
    if "urls" in df.columns:
        df["urls"] = df["urls"]
    else:
        df["urls"] = df.apply(
            lambda row: contains_url(row["subject"]) or contains_url(row["body"]),
            axis=1,
        ).astype(int)

    df["sender"] = df["sender"] if "sender" in df.columns else None
    df["sender_domain"] = df["sender"].apply(
        lambda x: x.split("@")[-1].lower() if pd.notna(x) else "unknown"
    )

    df["receiver"] = df["receiver"] if "receiver" in df.columns else None
    df["receiver_domain"] = df["receiver"].apply(
        lambda x: x.split("@")[-1].lower() if pd.notna(x) else "unknown"
    )

    df["same_domain"] = df.apply(
        lambda row: row["sender_domain"] == row["receiver_domain"], axis=1
    ).astype(int)

    df["names"] = df["sender"].apply(
        lambda x: x.split("@")[0].lower().split("<") if pd.notna(x) else "unknown"
    )
    df["claimed_vs_username"] = (
        df["names"]
        .apply(lambda row: similar(row[0], row[1]) if len(row) > 1 else 0)
        .astype(float)
    )

    df["subject"] = df["subject"].apply(clean_text)
    df["subject"] = df["subject"].apply(remove_stopwords)

    df["body"] = df["body"].apply(clean_text)
    df["body"] = df["body"].apply(remove_stopwords)

    return df[
        [
            "subject",
            "body",
            "label",
            "urls",
            "sender_domain",
            "receiver_domain",
            "same_domain",
            "claimed_vs_username",
        ]
    ]


def increase_url_weight(x):
    return x * 3


def increase_subject_weight(x):
    return x * 3.0


def increase_body_weight(x):
    return x * 3.5


def tackle():
    csv_files = glob.glob("../dataset/detail/*.csv")  # Adjust this path
    dataframes = [load_and_normalize_csv(f) for f in csv_files]
    print("Combining data...")
    df = pd.concat(dataframes, ignore_index=True)

    X = df[
        [
            "subject",
            "body",
            "urls",
            "sender_domain",
            "receiver_domain",
            "same_domain",
            "claimed_vs_username",
        ]
    ]
    y = df["label"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "subject",
                TfidfVectorizer(stop_words="english", max_features=300),
                "subject",
            ),
            ("body", TfidfVectorizer(stop_words="english", max_features=1000), "body"),
            ("same_domain", "passthrough", ["same_domain"]),
            ("claimed_vs_username", "passthrough", ["claimed_vs_username"]),
            ("urls", "passthrough", ["urls"]),
        ]
    )

    models = {
        "NaiveBayes": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
    }

    for name, clf in models.items():
        model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training...")
        model.fit(X_train, y_train)

        print("Predicting")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        print(f"{name} accuracy: {model.score(X_test, y_test):.4f}")
        print("Dumping model...")
        joblib.dump(model, f"models/{name}_phishing_model.pkl")
        evaluate_model(model, X_test, y_test, model_name=name)
