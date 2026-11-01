import streamlit as st
from utils.i18n import t, DEFAULT_LANGUAGE

st.set_page_config(page_title="Sleep & HRV", layout="wide")
# Widget runs outside individual pages to keep the choice when navigating.
st.sidebar.selectbox(
    "Language / Sprache", ["en", "de"],
    format_func=lambda code: {"en": "English", "de": "Deutsch"}[code],
    key="language", index=["en", "de"].index(DEFAULT_LANGUAGE),
)
st.set_page_config(page_title=t("Sleep & HRV"))
page = st.navigation([
    st.Page("utils/dashboard.py", title=t("Overview"), default=True, url_path="dashboard"),
    st.Page("views/1_Daten_Exploration.py", title=t("Data Exploration"), url_path="Daten_Exploration"),
    st.Page("views/2_Visualisierung.py", title=t("Visualization"), url_path="Visualisierung"),
    st.Page("views/4_ML_Ridge_Regression.py", title=t("Ridge Regression"), url_path="ML_Ridge_Regression"),
    st.Page("views/5_ML_Logistic_Regression.py", title=t("Logistic Regression"), url_path="ML_Logistic_Regression"),
])
page.run()
