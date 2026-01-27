import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# =============================
# App Config
# =============================
st.set_page_config(page_title="HRV Stress Classification")


st.title("HRV-Stressklassifikation mit logistischer Regression")

# =============================
# Load Data
# =============================
@st.cache_data
def load_data(version="v2"):
    return pd.read_csv("data/processed/data_final.csv")

df = load_data(version="2026-01-27")

# =============================
# Session State
# =============================
for key in [
    "logreg_model", "auc", "acc",
    "y_test", "y_pred", "y_prob",
    "q25", "q75"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================
# Feature / Target
# =============================
features = ["sleep_sum", "sleep_quality", "stepCount", "sportTime"]

# =============================
# Description
# =============================
st.markdown("""
### Modellbeschreibung

**Zielvariable (binär):**
- **low_hrv**
  - `1` → niedrige HRV (**hohes Stressniveau**)
  - `0` → hohe HRV (**niedriges Stressniveau**)

Die Zielvariable wird über Quantile gebildet:
- unteres 25 % (<= 35 ms) → **low_hrv = 1**
- oberes 25 %（>= 43 ms）→ **low_hrv = 0**
- mittlere 50 % werden ausgeschlossen

**Prädiktoren:**
- **sleep_sum** — Gesamte Schlafdauer (Minuten)
- **sleep_quality** — Schlafqualität-Score (0–100)
- **stepCount** — Anzahl Schritte pro Tag
- **sportTime** — Sportdauer (Minuten)

**Modell:**
- Logistische Regression (binäre Klassifikation)
- Standardisierung der Features
- Evaluation mit Accuracy & ROC-AUC
""")

# =============================
# Train Model Button
# =============================
st.subheader("Train Logistic Regression Model")

if st.button("Train Model"):
    with st.spinner("Training logistic regression..."):

        # --- Quantile ---
        q25 = df["mean_hrv"].quantile(0.25)
        q75 = df["mean_hrv"].quantile(0.75)

        # --- Binary dataset ---
        df_bin = df[
            (df["mean_hrv"] <= q25) |
            (df["mean_hrv"] >= q75)
        ].copy()

        df_bin["low_hrv"] = (df_bin["mean_hrv"] <= q25).astype(int)

        X = df_bin[features]
        y = df_bin["low_hrv"]

        # --- Train/Test ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # --- Pipeline ---
        logreg_model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000))
        ])

        logreg_model.fit(X_train, y_train)

        # --- Evaluation ---
        y_pred = logreg_model.predict(X_test)
        y_prob = logreg_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # --- Save state ---
        st.session_state["logreg_model"] = logreg_model
        st.session_state["acc"] = acc
        st.session_state["auc"] = auc
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["y_prob"] = y_prob
        st.session_state["q25"] = q25
        st.session_state["q75"] = q75

        st.success("Model trained and saved successfully.")

# =============================
# Results
# =============================
logreg_model = st.session_state["logreg_model"]

if logreg_model is not None:
    st.divider()
    st.subheader("Model Performance (Test Set)")

    col1, col2 = st.columns([2, 3])

    # ---- Metrics ----
    with col1:
        st.metric("Accuracy", f"{st.session_state['acc']:.2f}")
        st.metric("ROC-AUC", f"{st.session_state['auc']:.2f}")
        st.caption(
            "Positive Klasse: **low HRV (hoher Stress)**"
        )

    # ---- Confusion Matrix ----
    with col2:
        cm = confusion_matrix(
            st.session_state["y_test"],
            st.session_state["y_pred"]
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["High HRV", "Low HRV"]
        )
        disp.plot(ax=ax, cmap="Greys", colorbar=False)
        ax.set_title("Confusion Matrix – Logistic Regression")
        st.pyplot(fig)

    # ---- Classification Report ----
    with st.expander("Show classification report"):
        report = classification_report(
            st.session_state["y_test"],
            st.session_state["y_pred"],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(3))

# =============================
# User Prediction
# =============================
if logreg_model is not None:
    st.divider()
    st.subheader("Predict Stress Level (User Input)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sleep_hours = st.number_input(
            "Sleep duration (hours)", 0.0, 24.0, 7.0, 0.25
        )
    with col2:
        sleep_quality = st.slider("Sleep quality", 0.0, 100.0)
    with col3:
        step_count = st.number_input("Step count", min_value=0, value=8000)
    with col4:
        sport_time = st.number_input("Sport time (min)", min_value=0.0, value=30.0)

    if st.button("Predict Stress Level"):
        user_X = pd.DataFrame([[
            sleep_hours * 60,
            sleep_quality,
            step_count,
            sport_time
        ]], columns=features)

        prob_low_hrv = logreg_model.predict_proba(user_X)[0, 1]
        pred_class = int(prob_low_hrv >= 0.5)

        if pred_class == 1:
            st.error(
                f"⚠️ **High stress predicted**  \n"
                f"Probability (low HRV): **{prob_low_hrv:.2f}**"
            )
        else:
            st.success(
                f"✅ **Low stress predicted**  \n"
                f"Probability (low HRV): **{prob_low_hrv:.2f}**"
            )

else:
    st.info("Please train the model first.")
