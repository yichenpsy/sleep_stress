from utils.i18n import t, column_label, display_data
import streamlit as st
import pandas as pd
from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = "data/processed/data_final.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader(t('Dataset Overview'))
st.write(t("{rows} rows, {columns} columns", rows=len(df), columns=len(df.columns)))
st.dataframe(display_data(df.head()))

st.subheader(t('Statistics'))
st.write(df.describe())
