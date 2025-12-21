import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/sleep_clean.csv")

df = load_data()

st.subheader("Datensatz Übersicht")
st.write(f"{len(df)} Zeilen, {len(df.columns)} Spalten")
st.dataframe(df.head())

st.subheader("Statistiken")
st.write(df.describe())
