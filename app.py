import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sleep & HRV", layout="wide")
st.title("🛌 Sleep & Stress Dashboard")

st.markdown("""
Diese App analysiert meinen Schlaf, HRV und Aktivität
und zeigt Zusammenhänge und Vorhersagen.
👈 Wähle links eine Seite!
""")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data_final.csv")

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Tage", len(df))
col2.metric("Ø HRV", round(df["mean_hrv"].mean(),1))
col3.metric("Ø Schlaf", round(df["sleep_quality"].mean(),1))

st.dataframe(df.head())
