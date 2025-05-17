from phisher_man.util import contains_url, similar, clean_text, remove_stopwords
from pprint import pprint
import joblib
import numpy as np
import pandas as pd
import re
from tabulate import tabulate

MODEL = "RF"


def get_feature_names(preprocessor, input_df):
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        if transformer == "passthrough":
            if isinstance(columns, str):
                columns = [columns]
            feature_names.extend(columns)
            continue
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
        else:
            last_step = transformer
        if hasattr(last_step, "get_feature_names_out"):
            try:
                fn = last_step.get_feature_names_out(columns)
            except:
                fn = last_step.get_feature_names_out()
            feature_names.extend(fn)
        else:
            if isinstance(columns, str):
                columns = [columns]
            for col in columns:
                feature_names.append(f"{name}__{col}")
    return feature_names


def explain_contributions(feature_names, contributions, human_readbale=True):
    explanation = []

    # Get top 5 most influential features
    top_indices = np.argsort(np.abs(contributions))[::-1][:5]

    for idx in top_indices:
        if idx >= len(feature_names):
            continue
        name = feature_names[idx]
        contrib = contributions[idx]

        # Determine meaning
        if (
            name.startswith("body__")
            or name.startswith("subject__")
            or name.startswith("text_combined__")
        ):
            token = re.sub(r"^(.*)__", "", name)
            reason = f"word '{token}'"
        elif "urls" in name:
            token = name.split("=")[-1]
            reason = "presence of a URL"
        elif "sender_domain" in name:
            token = name.split("=")[-1]
            reason = f"sender domain '{token}'"
        elif "receiver_domain" in name:
            token = name.split("=")[-1]
            reason = f"receiver domain '{token}'"
        elif "same_domain" in name:
            token = name.split("=")[-1]
            reason = f"'{token}'"
        elif "claimed_vs_username" in name:
            token = name.split("=")[-1]
            reason = f"'{token}'"
        else:
            token = name.split("=")[-1]
            reason = f"feature '{name}'"

        polarity = "increased" if contrib > 0 else "decreased"
        if human_readbale:
            explanation.append(
                f"- {reason} {polarity} likelihood (weight: {contrib:.4f})"
            )
        else:
            explanation.append([token, polarity, contrib])

    return explanation


def predict_email(
    sender,
    receiver,
    subject,
    body,
    model="./phishing_model.pkl",
    model_type="LogisticRegression",
    human_readbale=True,
):
    if human_readbale:
        output = []
    else:
        output = {}
    model = joblib.load(model)

    sender_domain = sender.split("@")[-1].lower()
    receiver_domain = receiver.split("@")[-1].lower()
    urls = int(contains_url(body) or contains_url(subject))
    text_combined = f"{sender or ''} {subject or ''} {body or ''}"
    same_domain = 0

    names = sender.split("@")[0].lower().split("<")
    # output.append(names)
    claimed_vs_username = similar(names[0], names[-1]) if len(names) > 1 else 0
    # output.append(claimed_vs_username)
    # output.append(same_domain)

    if sender_domain == receiver_domain:
        same_domain = 1

    subject = clean_text(subject)
    subject = remove_stopwords(subject)

    body = clean_text(body)
    body = remove_stopwords(body)

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
                "text_combined": text_combined,
            }
        ]
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    prediction = model.predict(email_df)[0]
    probability = model.predict_proba(email_df)[0][1]
    X_transformed = preprocessor.transform(email_df)

    result = "PHISHING" if prediction == 1 else "LEGITIMATE"
    if human_readbale:
        output.append(f"📨 Prediction: {result}")
        output.append(f"🔍 Confidence: {probability:.2f}")
    else:
        output["prediciton"] = result
        output["confidence"] = probability

    # Explain decision
    feature_names = get_feature_names(preprocessor, email_df)

    # MultinomialNB: use log-probability differences

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

    if human_readbale:
        output.append(f"🔢 Number of features: {len(feature_names)}")
        output.append(f"🔢 Contributions shape: {contributions.shape}")
        output.append("💡 Top reasons for prediction:")
        output.append(f"Subject: {subject}")
        output.extend(explain_contributions(feature_names, contributions))
        output.append("")
    else:
        output["subject"] = subject
        output["contributions"] = explain_contributions(
            feature_names, contributions, human_readbale=False
        )
    return output


def bob_all():
    human_readable = False
    models = [
        "NaiveBayes",
        "LogisticRegression",
        "RandomForest",
    ]

    output = []
    for model in models:
        output.append(
            run_predictions_from_csv(
                f"./models/{model}_phishing_model.pkl",
                model,
                human_readable=human_readable,
            )
        )

    if human_readable:
        output = [list(row) for row in zip(*output)]
        print(tabulate(output, models, tablefmt="orgtbl"))
    else:
        total = 0
        phishing_count = 0
        legit_count = 0
        each_correct = [0, 0, 0]
        each_phishing = [0, 0, 0]
        each_legit = [0, 0, 0]
        for x in range(len(output[0])):
            confidence = 0
            for i in range(3):
                email = output[i][x]
                # pprint(email)
                confidence = email["confidence"]

                confidence = confidence
                prediciton = "LEGITIMATE"
                if confidence >= 0.5:
                    prediciton = "PHISHING"
                    each_phishing[i] += 1
                else:
                    each_legit[i] += 1

                if email["expected"] == prediciton:
                    each_correct[i] += 1
            # print(f"""
            # confidence: {confidence}
            # expected:   {output[0][x]['expected']}
            # prediciton: {prediciton}
            # NB: {output[0][x]['prediciton']}
            # LR: {output[1][x]['prediciton']}
            # RF: {output[2][x]['prediciton']}
            # """)

            if output[0][x]["expected"] == "PHISHING":
                phishing_count += 1
            else:
                legit_count += 1

            total += 1

        (each_correct.insert(0, "Correct"),)
        (each_legit.insert(0, "Legitimate"),)
        (each_phishing.insert(0, "Phishing"),)
        tab_correct = [
            [
                "Prediction Counts",
                "Naive Bayes",
                "Logistic Regression",
                "Random Forest",
            ],
            each_correct,
            each_legit,
            each_phishing,
        ]
        print(tabulate(tab_correct, tablefmt="grid"))
        print(f"Total emails tested: {total}")
        print(f"Total Legitimate emails tested: {legit_count}")
        print(f"Total Phishing emails tested: {phishing_count}")
        # pprint(output[0])


def run_predictions_from_csv(model, model_type, human_readable=True):
    df = pd.read_csv("./emails.csv")
    all_outputs = []

    for _, row in df.iterrows():
        expected = row["expected"]
        sender = row["sender"]
        receiver = row["receiver"]
        subject = row["subject"]
        body = row["body"]

        if human_readable:
            prediction_output = []
        else:
            prediction_output = {}

        # Add expected label
        if human_readable:
            all_outputs.append(f"Expected: {expected}")

        # Run prediction
        prediction_output = predict_email(
            sender=sender,
            receiver=receiver,
            subject=subject,
            body=body,
            model=model,
            model_type=model_type,
            human_readbale=human_readable,
        )

        if not human_readable:
            prediction_output["expected"] = expected.upper()
        # Collect results
        if human_readable:
            all_outputs.extend(prediction_output)
        else:
            all_outputs.append(prediction_output)

    return all_outputs


def bob(model="./phishing_model.pkl", model_type="LogisticRegression"):
    output = []
    output.append("Expected: Phising")
    output.extend(
        predict_email(
            sender="fake@phishy.com",
            receiver="someone@gmail.com",
            subject="Update your billing info",
            body="Click here to reset your account credentials. http://trustme.com",
            model=model,
            model_type=model_type,
        )
    )

    output.append("Expected: Phising")
    output.extend(
        predict_email(
            sender="real@normal.com",
            receiver="someone@gmail.com",
            # subject='Cos720 Module Checkin update account',
            subject="Cos720 Checkin update account",
            body="Click here to reset your account credentials. http://trustme.com",
            model=model,
            model_type=model_type,
        )
    )

    output.append("Expected: Legitimate")
    output.extend(
        predict_email(
            sender="real@gmail.com",
            receiver="someone@gmail.com",
            subject="Cos720 Module Checkin",
            body="Good afternoon sir I wanted to find out if you could email me the details of the module",
            model=model,
            model_type=model_type,
        )
    )

    output.append("Expected: Either Or")
    output.extend(
        predict_email(
            sender="fake@phisy.com",
            receiver="someone@gmail.com",
            subject="Update your billing info",
            body="Good afternoon sir I wanted to find out if you could email me the details of the module",
            model=model,
            model_type=model_type,
        )
    )

    output.append("Expected: Either Or")
    output.extend(
        predict_email(
            sender="fake@phisy.com",
            receiver="someone@gmail.com",
            subject="Update your billing info",
            body="Good afternoon sir I wanted to find out if you could email me the details of the module",
            model=model,
            model_type=model_type,
        )
    )

    output.append("Expected: Phishing")
    output.extend(
        predict_email(
            sender="Morkel Person <lisabake25@gmail.com>",
            receiver="caden@southafrica.co.za",
            subject="John Doe help please",
            body="""
        How are you doing  Ivan,
        I am currently in a conference
        and I want to you take care of a very task.
        Send your mobile number and wait for my Whatsapp
        text.............
        
        Best Greetings
        """,
            model=model,
            model_type=model_type,
        )
    )
    return output
