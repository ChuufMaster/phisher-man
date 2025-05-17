import numpy as np
import re


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


def explain_contributions(feature_names, contributions, human_readable=False):
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
        if human_readable:
            explanation.append(
                f"- {reason} {polarity} likelihood (weight: {contrib:.4f})"
            )
        else:
            explanation.append([token, polarity, contrib])

    return explanation
