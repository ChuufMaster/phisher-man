import joblib
import logging
import pandas as pd
import numpy as np
from app.utils.preprocess import clean_text, contains_url, remove_stopwords, similar

# GLOBAL_MODEL_TYPE = "LogisticRegression"
GLOBAL_MODEL_TYPE = "NaiveBayes"
# GLOBAL_MODEL_TYPE = "RandomForest"

logger = logging.getLogger(__name__)


def predict(email_input: dict):
    model_type = GLOBAL_MODEL_TYPE
    model = joblib.load(f"models/{model_type}_phishing_model.pkl")
    sender = email_input["sender"]
    receiver = email_input["receiver"]
    subject = clean_text(remove_stopwords(email_input["subject"]))
    body = clean_text(remove_stopwords(email_input["body"]))

    sender_domain = sender.split("@")[-1].lower()
    receiver_domain = receiver.split("@")[-1].lower()
    urls = int(contains_url(body) or contains_url(subject))
    same_domain = int(sender_domain == receiver_domain)
    names = sender.split("@")[0].lower().split("<")
    claimed_vs_username = similar(names[0], names[-1]) if len(names) > 1 else 0

    email_df = pd.DataFrame(
        [
            {
                "body": body,
                "subject": subject,
                "sender_domain": sender_domain,
                "receiver_domain": receiver_domain,
                "same_domain": same_domain,
                "urls": urls,
                "claimed_vs_username": claimed_vs_username,
                "text_combined": f"{sender} {subject} {body}",
            }
        ]
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    X_transformed = preprocessor.transform(email_df)
    prediction = model.predict(email_df)[0]
    probability = model.predict_proba(email_df)[0][1]

    if model_type == "NaiveBayes":
        log_prob_diff = (
            classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
        )
        contributions = X_transformed.toarray()[0] * log_prob_diff
    elif model_type == "LogisticRegression":
        weights = classifier.coef_[0]
        contributions = X_transformed.toarray()[0] * weights
    elif model_type == "RandomForest":
        contributions = classifier.feature_importances_

    from app.services.explain import get_feature_names, explain_contributions

    feature_names = get_feature_names(preprocessor, email_df)
    reasons = explain_contributions(feature_names, contributions)

    return {
        "prediction": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "confidence": round(probability, 2),
        "reasons": reasons,
    }
