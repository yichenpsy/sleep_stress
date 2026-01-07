import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "processed" / "data_final.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader("Datensatz Übersicht")
st.write(f"{len(df)} Zeilen, {len(df.columns)} Spalten")
st.dataframe(df.head())

st.subheader("Statistiken")
st.write(df.describe())
