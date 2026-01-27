import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data exploration")

@st.cache_data
def load_data(version="v2"):
    return pd.read_csv("data/processed/data_final.csv")

df = load_data(version="2026-01-27")

# Sidebar Filter
df["date"] = pd.to_datetime(df["date"])

start_date, end_date = st.slider(
    "Date",
    min_value=df["date"].min().to_pydatetime(),
    max_value=df["date"].max().to_pydatetime(),
    value=(
        df["date"].min().to_pydatetime(),
        df["date"].max().to_pydatetime()
    )
)

df = df[
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
]

st.metric("Filtered Days", len(df))
st.dataframe(df)

feature = st.selectbox(
    "variable",
    ["mean_hrv", "sleep_quality", "sleep_sum_hours", "stepCount", "sportTime"]
)

# ---------------- Filtering logic ----------------
df_plot = df


# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(15, 6))

ax.bar(df_plot["date"], df_plot[feature])
ax.set_xlabel("Date")
ax.set_ylabel(feature)
ax.set_title(f"{feature} over time")

plt.xticks(rotation=45)
st.pyplot(fig)


