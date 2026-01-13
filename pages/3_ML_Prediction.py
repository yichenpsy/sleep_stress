import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("HRV Prediction")

df = pd.read_csv("data/processed/data_final.csv")

if st.button("Train Model"):
    X = df[["sleep_quality","stepCount","sportTime"]]
    y = df["mean_hrv"]

    model = LinearRegression()
    model.fit(X,y)

    st.session_state.model = model
    st.success("Model trained!")

if "model" in st.session_state:
    sleep = st.number_input("Sleep Quality", 0.0,10.0,7.0)
    steps = st.number_input("Steps",0,20000,8000)
    sport = st.number_input("Sport Minutes",0,180,30)

    if st.button("Predict HRV"):
        pred = st.session_state.model.predict([[sleep,steps,sport]])
        st.metric("Predicted HRV", round(pred[0],2))
