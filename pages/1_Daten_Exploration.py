import streamlit as st
import pandas as pd

st.title("Daten Exploration")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data_final.csv")

df = load_data()

# Sidebar Filter
min_steps, max_steps = st.sidebar.slider(
    "Schritte",
    int(df.stepCount.min()),
    int(df.stepCount.max()),
    (2000, 12000)
)

df = df[(df.stepCount >= min_steps) & (df.stepCount <= max_steps)]

st.metric("Gefilterte Tage", len(df))
st.dataframe(df)
