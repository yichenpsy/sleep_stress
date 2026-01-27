import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# st.title("HRV Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data_final.csv")

df = load_data()

## Train model
X = df[["sleep_sum","sleep_quality","stepCount","sportTime"]]
y = df["mean_hrv"]

if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train,y_train)
    st.session_state.model = model

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    pipe.fit(X_train, y_train)

    beta = pipe.named_steps["lr"].coef_  # standardized coefficients
    effect = np.abs(beta)
    coefficients = pipe.named_steps["lr"].coef_

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_,
        "Beta (standardized)": beta,
        "Effect score |Beta|": effect,
        "Effect %": effect / effect.sum() * 100
    }).sort_values("Effect score |Beta|", ascending=False)

    st.dataframe(coef_df)

    #### Model Evaluation

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("Model Evaluation")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", round(r2, 3))
    col2.metric("MAE", round(mae, 2))
    col3.metric("RMSE", round(rmse, 2))

    st.subheader("Predicted vs Actual HRV")

    fig, ax = plt.subplots()

    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    ax.set_xlabel("Actual HRV")
    ax.set_ylabel("Predicted HRV")
    ax.set_title("Regression Performance")

    st.pyplot(fig)


####


if "model" in st.session_state:
    st.subheader("Predict HRV")

    sleep_sum = st.number_input("Sleep Duration (h)", 0.0, 12.0, 7.5)
    sleep_quality = st.number_input("Sleep Quality", 0.0, 10.0, 7.0)
    steps = st.number_input("Steps", 0, 20000, 8000)
    sport = st.number_input("Sport Minutes", 0, 180, 30)

    if st.button("Predict HRV"):
        X_new = pd.DataFrame([{
            "sleep_sum": sleep_sum,
            "sleep_quality": sleep_quality,
            "stepCount": steps,
            "sportTime": sport
        }])
        pred = st.session_state.model.predict(X_new)
        st.metric("Predicted HRV", round(pred[0], 2))



