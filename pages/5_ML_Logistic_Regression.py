import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.title("HRV Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data_final.csv")
df = load_data()

low_threshold  = df["mean_hrv"].quantile(0.33)
high_threshold = df["mean_hrv"].quantile(0.66)



low_thr  = df["mean_hrv"].quantile(0.33)
high_thr = df["mean_hrv"].quantile(0.66)

df["stress_label"] = np.nan
df.loc[df["mean_hrv"] <= low_thr,  "stress_label"] = 1  # high stress
df.loc[df["mean_hrv"] >= high_thr, "stress_label"] = 0  # low stress

df_clf = df.dropna(subset=["stress_label"])

X = df_clf[["sleep_sum", "sleep_quality", "stepCount", "sportTime"]]
y = df_clf["stress_label"]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

if st.button("Train Stress Classifier"):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    pipe.fit(X_train, y_train)
    st.session_state.model = pipe

    # Evaluation
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.metric("Accuracy", round(acc, 3))

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    coef = pipe.named_steps["clf"].coef_[0]

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coef,
        "Effect strength |coef|": np.abs(coef)
    }).sort_values("Effect strength |coef|", ascending=False)

    st.subheader("Feature Importance (Stress Prediction)")
    st.dataframe(coef_df)

if "model" in st.session_state:
    st.subheader("Predict Stress Level")

    sleep_sum = st.number_input("Sleep Duration (h)", 0.0, 12.0, 7.5)
    sleep_quality = st.number_input("Sleep Quality", 0.0, 10.0, 7.0)
    steps = st.number_input("Steps", 0, 20000, 8000)
    sport = st.number_input("Sport Minutes", 0, 180, 30)

    if st.button("Predict Stress"):
        X_new = pd.DataFrame([{
            "sleep_sum": sleep_sum,
            "sleep_quality": sleep_quality,
            "stepCount": steps,
            "sportTime": sport
        }])

        prob = st.session_state.model.predict_proba(X_new)[0]
        pred = st.session_state.model.predict(X_new)[0]

        st.metric(
            "Predicted Stress Level",
            "High Stress" if pred == 1 else "Low Stress"
        )

        st.write(f"Probability High Stress: {prob[1]:.2f}")
