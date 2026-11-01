from utils.i18n import t, column_label, display_data
import streamlit as st
import pandas as pd
from pathlib import Path


st.markdown(
    """
    <style>
    .block-container {
        padding: 2.5rem 4rem 2.5rem 4rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(t('Sleep, Sport & Stress Dashboard'))

st.markdown(t('This app analyzes my sleep, HRV, and activity from 2022 to 2025 and shows relationships and predictions.'))

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parents[1]
    data_path = BASE_DIR / "data" / "processed" / "data_final.csv"
    return pd.read_csv(data_path)

df = load_data()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric(t('Days'), len(df))
col2.metric(t('Average HRV (lower values indicate higher stress)'), round(df["mean_hrv"].mean(),1))
col3.metric(t('Average Sleep Quality [0, 100]'), round(df["sleep_quality"].mean(),1))
col4.metric(t('Average Sleep Duration (h)'), round(df["sleep_sum"].mean()/60,1))
col5.metric(t('Average Daily Steps'), round(df["stepCount"].mean(),1))
col6.metric(t('Average Daily Sport Time (min)'), round(df["sportTime"].mean(),1))

st.dataframe(display_data(df))
