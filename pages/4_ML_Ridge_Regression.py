import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error


# =============================
# App Config
# =============================
st.set_page_config(page_title="HRV Prediction")


st.title("HRV-Vorhersage mit Ridge-Regression")


# =============================
# Load Data
# =============================
@st.cache_data
def load_data(version="v2"):
    return pd.read_csv("data/processed/data_final.csv")

df = load_data(version="2026-01-27")


# =============================
# Init Session State
# =============================
for key in ["ridge_model", "r2", "mae", "alpha", "y_test", "y_pred"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =============================
# Feature / target
# =============================
features = ["sleep_sum", "sleep_quality", "stepCount", "sportTime"]
X = df[features]
y = df["mean_hrv"]


# =============================
# Train Model Button
# =============================
st.markdown("""
### Modellbeschreibung

**Zielvariable (Target):**
- **mean_hrv** — Durchschnittliche Herzratenvariabilität (HRV) als Indikator für Stress und Erholung  
  *(Höhere HRV = geringeres Stressniveau)*

**Prädiktoren (Features):**
- **sleep_sum** — Gesamte Schlafdauer (in Minuten)
- **sleep_quality** — Normalisierter Schlafqualitäts-Score (0–1)
- **stepCount** — Anzahl der täglichen Schritte
- **sportTime** — Sportdauer pro Tag (in Minuten)

**Modell:**
- Ridge Regression (lineares Regressionsmodell mit L2-Regularisierung)
- Ziel: Vorhersage der HRV basierend auf Schlaf- und Aktivitätsverhalten
""")


st.subheader("Train Ridge Regression Model")
if st.button("Train Model"):
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        ridge_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
        ])

        ridge_model.fit(X_train, y_train)

        # Evaluation
        y_pred = ridge_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        alpha = ridge_model.named_steps["ridge"].alpha_

        # =============================
        # Save to session_state
        # =============================
        st.session_state["ridge_model"] = ridge_model
        st.session_state["r2"] = r2
        st.session_state["mae"] = mae
        st.session_state["alpha"] = alpha
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred

        st.success("Model trained and saved successfully.")


# =============================
# Show Model Results (STATE-BASED)
# =============================
ridge_model = st.session_state["ridge_model"]

if ridge_model is not None:
    st.divider()
    st.subheader("Model Performance (Test Set)")

    col1, col2 = st.columns([2, 3])

    # ---- LEFT: Metrics ----
    with col1:
        st.metric("R² (Test)", f"{st.session_state['r2']:.2f}")
        st.metric("MAE (Test)", f"{st.session_state['mae']:.2f}")
        st.metric("Best α", f"{st.session_state['alpha']:.2f}")

    # ---- RIGHT: Plot ----
    with col2:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle="--"
        )
        ax.set_xlabel("Actual mean_hrv")
        ax.set_ylabel("Predicted mean_hrv")
        ax.set_title("Actual vs Predicted (Test set)")

        ax.text(
            0.05, 0.95,
            f"$R^2$ = {st.session_state['r2']:.2f}\n"
            f"MAE = {st.session_state['mae']:.2f}\n"
            f"Best α = {st.session_state['alpha']:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.8)
        )

        st.pyplot(fig)


# =============================
# User Input Prediction
# =============================
if ridge_model is not None:
    st.divider()
    st.subheader("Predict HRV (User Input)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sleep_hours = st.number_input(
            "Sleep duration (hours)",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.25
        )
    with col2:
        sleep_quality = st.slider("Sleep quality", 0.0, 1.0, 0.7)
    with col3:
        step_count = st.number_input("Step count", min_value=0, value=8000)
    with col4:
        sport_time = st.number_input("Sport time (min)", min_value=0.0, value=30.0)

    if st.button("Predict HRV"):
        sleep_sum_minutes = sleep_hours * 60  # hours → minutes

        user_X = pd.DataFrame([[
            sleep_sum_minutes,
            sleep_quality,
            step_count,
            sport_time
        ]], columns=features)

        prediction = ridge_model.predict(user_X)[0]
        st.success(f"Predicted mean HRV: **{prediction:.2f}**")

else:
    st.info("Please train the model first to enable prediction.")
