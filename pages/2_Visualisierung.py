import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 HRV Visualisierung")

df = pd.read_csv("data/processed/data_final.csv")

feature = st.selectbox("Feature", ["mean_hrv","sleep_quality","stepCount"])

fig, ax = plt.subplots()
df[feature].hist(ax=ax, bins=20)
st.pyplot(fig)
