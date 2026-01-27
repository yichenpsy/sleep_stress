import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sleep & HRV", layout="wide")
st.title("Sleep, Sport & Stress Dashboard")

st.markdown("""
Diese App analysiert meinen Schlaf, HRV und Aktivität
und zeigt Zusammenhänge und Vorhersagen.
""")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data_final.csv")

df = load_data()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Day", len(df))
col2.metric("Ø HRV", round(df["mean_hrv"].mean(),1))
col3.metric("Ø Sleep Quality [0, 100]", round(df["sleep_quality"].mean(),1))
col4.metric("Ø Sleep Duration (h)", round(df["sleep_sum"].mean()/60,1))
col5.metric("Ø Step count per day", round(df["stepCount"].mean(),1))
col6.metric("Ø Sport time per day (min)", round(df["sportTime"].mean(),1))

st.dataframe(df.head())
