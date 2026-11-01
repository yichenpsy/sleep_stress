from utils.i18n import t, column_label, display_data
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title(t('Data Exploration'))

@st.cache_data
def load_data(version="v2"):
    return pd.read_csv("data/processed/data_final.csv")

df = load_data(version="2026-01-27")

# Sidebar Filter
df["date"] = pd.to_datetime(df["date"])

start_date, end_date = st.slider(
    t('Date'),
    min_value=df["date"].min().to_pydatetime(),
    max_value=df["date"].max().to_pydatetime(),
    value=(
        df["date"].min().to_pydatetime(),
        df["date"].max().to_pydatetime()
    ), key="exploration_dates"
)

df = df[
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
]

st.metric(t('Filtered Days'), len(df))
st.dataframe(display_data(df))

feature = st.selectbox(
    t('Variable'),
    ["mean_hrv", "sleep_quality", "sleep_sum_hours", "stepCount", "sportTime"],
    format_func={name: column_label(name) for name in df.columns}.__getitem__, key="1_Daten_Exploration_variable"
)

# ---------------- Filtering logic ----------------
df_plot = df


# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(15, 6))

ax.bar(df_plot["date"], df_plot[feature])
ax.set_xlabel(t('Date'))
ax.set_ylabel(column_label(feature))
ax.set_title(t('{feature} over time', feature=column_label(feature)))

plt.xticks(rotation=45)
st.pyplot(fig)


