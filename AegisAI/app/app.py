import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Helpers for pretty printing
# ----------------------------
def pretty_duration(days):
    if days is None or pd.isna(days):
        return "Unknown"
    if days <= 0:
        return "Same-day event"
    if int(days) == 1:
        return "1 day"
    return f"{int(days)} days"


def pretty_region(region_num):
    return f"FEMA Region {int(region_num)}" if pd.notna(region_num) else "Unknown region"


def predict_with_confidence(model, row_dict):
    X_one = pd.DataFrame([row_dict])
    probs = model.predict_proba(X_one)[0]
    classes = model.classes_
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    prob_table = dict(zip(classes, probs))
    return pred_class, confidence, prob_table


def top_reasons_from_inputs(inputs, topn=3):
    reasons = []
    dec_type = inputs["declaration_type"]
    inc_type = inputs["incident_type"]

    if dec_type == "DR":
        reasons.append("Declaration type is DR (Major Disaster), a strong severity driver")
    elif dec_type == "EM":
        reasons.append("Declaration type is EM (Emergency), strongly linked to elevated severity")

    reasons.append(f"Incident year is {int(inputs['incident_year'])}, which influences severity patterns")
    reasons.append(f"Seasonality: month {int(inputs['incident_month'])} / quarter {int(inputs['incident_quarter'])}")

    dur = inputs.get("disaster_duration_days", 0)
    if pd.isna(dur) or dur <= 0:
        reasons.append("Duration is same-day; severity comes mostly from declaration + incident type")
    else:
        reasons.append(f"Duration is {int(dur)} days, which increases severity likelihood")

    reasons.append(f"Incident type '{inc_type}' contributes to severity risk profile")

    return reasons[:topn]


def generate_explanation(model_pred, conf, prob_table, inputs):
    duration_txt = pretty_duration(inputs.get("disaster_duration_days"))
    region_txt = pretty_region(inputs.get("region"))
    time_txt = f"{int(inputs['incident_month'])}/{int(inputs['incident_year'])}"

    probs_txt = ", ".join([f"{k}: {v:.2f}" for k, v in prob_table.items()])
    reasons = top_reasons_from_inputs(inputs, topn=3)
    reason_block = "\n".join([f"- {r}" for r in reasons])

    if model_pred == "High":
        action = "Activate emergency operations center, mobilize resources, and prioritize public alerts."
    elif model_pred == "Medium":
        action = "Increase monitoring, pre-position supplies, and prepare shelters and response teams."
    else:
        action = "Routine monitoring and local response readiness."

    return f"""
Disaster Severity Prediction: {model_pred}  (confidence: {conf:.2f})

Key facts:
- Declaration type: {inputs['declaration_type']}
- Incident type: {inputs['incident_type']}
- Duration: {duration_txt}
- Time: {time_txt}
- Region: {region_txt}
- State: {inputs['state']}

Model probabilities:
{probs_txt}

Top reasons (model-informed):
{reason_block}

Recommended action:
{action}
""".strip()


# ----------------------------
# Load + Train (cached)
# ----------------------------
@st.cache_data
def load_data():
    base = Path(__file__).resolve().parents[1]  # AegisAI/
    df = pd.read_csv(base / "data" / "us_disaster_declarations.csv")

    # Drop sparse / metadata columns (only if they exist)
    drop_cols = ["last_ia_filing_date", "designated_incident_types", "hash", "id"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Datetime conversion
    for col in ["declaration_date", "incident_begin_date", "incident_end_date", "disaster_closeout_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Severity score + label
    sev_cols = ["ih_program_declared", "ia_program_declared", "pa_program_declared", "hm_program_declared"]
    df["severity_score"] = df[sev_cols].sum(axis=1)

    def sev_label(score):
        if score <= 1:
            return "Low"
        elif score == 2:
            return "Medium"
        else:
            return "High"

    df["severity_label"] = df["severity_score"].apply(sev_label)

    # Duration + time features
    df["disaster_duration_days"] = (df["incident_end_date"] - df["incident_begin_date"]).dt.days
    df["disaster_duration_days"] = df["disaster_duration_days"].fillna(0).clip(lower=0)

    df["incident_year"] = df["incident_begin_date"].dt.year
    df["incident_month"] = df["incident_begin_date"].dt.month
    df["incident_quarter"] = df["incident_begin_date"].dt.quarter

    # Keep usable rows
    df = df.dropna(subset=["incident_year", "incident_month", "incident_quarter"])

    return df


@st.cache_resource
def train_model(df):
    X = df[
        [
            "state",
            "region",
            "incident_type",
            "declaration_type",
            "disaster_duration_days",
            "incident_year",
            "incident_month",
            "incident_quarter",
        ]
    ]
    y = df["severity_label"]

    categorical_cols = ["state", "incident_type", "declaration_type"]
    numeric_cols = ["region", "disaster_duration_days", "incident_year", "incident_month", "incident_quarter"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )),
        ]
    )
    model.fit(X_train, y_train)
    return model


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AegisAI – Disaster Severity Demo", layout="wide")

st.title("AegisAI – Disaster Severity Prediction")
st.caption("Predicts disaster severity (Low / Medium / High) with confidence and an explainable briefing.")

df = load_data()
model = train_model(df)

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")

    state = st.selectbox("State", sorted(df["state"].dropna().unique().tolist()))
    region = st.selectbox("FEMA Region", sorted(df["region"].dropna().unique().tolist()))
    incident_type = st.selectbox("Incident Type", sorted(df["incident_type"].dropna().unique().tolist()))
    declaration_type = st.selectbox("Declaration Type", sorted(df["declaration_type"].dropna().unique().tolist()))

    incident_year = st.slider(
        "Incident Year",
        int(df["incident_year"].min()),
        int(df["incident_year"].max()),
        2017
    )
    incident_month = st.slider("Incident Month", 1, 12, 8)
    incident_quarter = (incident_month - 1) // 3 + 1

    disaster_duration_days = st.number_input("Disaster Duration (days)", min_value=0, max_value=365, value=0)

    inputs = {
        "state": state,
        "region": int(region),
        "incident_type": incident_type,
        "declaration_type": declaration_type,
        "disaster_duration_days": float(disaster_duration_days),
        "incident_year": int(incident_year),
        "incident_month": int(incident_month),
        "incident_quarter": int(incident_quarter),
    }

    # Demo magic: load a real FEMA record
    if st.button("Load Random Real Example"):
        sample = df.sample(1).iloc[0]

        inputs["state"] = sample["state"]
        inputs["region"] = int(sample["region"])
        inputs["incident_type"] = sample["incident_type"]
        inputs["declaration_type"] = sample["declaration_type"]
        inputs["disaster_duration_days"] = float(sample["disaster_duration_days"])
        inputs["incident_year"] = int(sample["incident_year"])
        inputs["incident_month"] = int(sample["incident_month"])
        inputs["incident_quarter"] = int(sample["incident_quarter"])

        st.success("Loaded a real FEMA disaster example. Now click Predict Severity.")

    run_btn = st.button("Predict Severity")

with right:
    st.subheader("Output")

    if run_btn:
        pred, conf, probs = predict_with_confidence(model, inputs)
        explanation = generate_explanation(pred, conf, probs, inputs)

        severity_color = {
            "High": "red",
            "Medium": "orange",
            "Low": "green"
        }[pred]

        st.markdown(
            f"<h2 style='color:{severity_color}'>Severity: {pred}</h2>",
            unsafe_allow_html=True
        )

        st.metric("Confidence", f"{int(conf * 100)}%")

        if conf < 0.5:
            st.warning("⚠️ Prediction confidence is moderate. Use with caution.")
        elif conf < 0.75:
            st.info("ℹ️ Prediction confidence is good.")
        else:
            st.success("✅ High confidence prediction.")

        prob_df = pd.DataFrame({
            "Class": list(probs.keys()),
            "Probability": list(probs.values())
        })

        st.bar_chart(prob_df.set_index("Class"))

        st.text_area("AI Briefing", value=explanation, height=320)

    else:
        st.info("Set inputs on the left and click **Predict Severity**.")
