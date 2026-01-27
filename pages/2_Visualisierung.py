import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Visualisation")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data_final.csv")


df = load_data()
df["date"] = pd.to_datetime(df["date"])

# ---------------- Feature selection ----------------
feature = st.selectbox(
    "variable",
    ["mean_hrv", "sleep_quality", "sleep_sum", "stepCount", "sportTime"]
)

# ---------------- Time selection (ON PAGE) ----------------
# --- One row: Year | Month ---
col1, col2 = st.columns(2)

with col1:
    selected_year = st.selectbox(
        "Year",
        ["All"] + sorted(df["date"].dt.year.unique().tolist())
    )

with col2:
    if selected_year != "All":
        df_year = df[df["date"].dt.year == selected_year]
        selected_month = st.selectbox(
            "Month",
            ["All"] + sorted(df_year["date"].dt.month.unique().tolist())
        )
    else:
        selected_month = "All"
        st.selectbox("Month", ["All"], disabled=True)


# ---------------- Filtering logic ----------------
df_plot = df

if selected_year != "All":
    df_plot = df_plot[df_plot["date"].dt.year == selected_year]

if selected_month != "All":
    df_plot = df_plot[df_plot["date"].dt.month == selected_month]

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(15, 6))

ax.bar(df_plot["date"], df_plot[feature])
ax.set_xlabel("Date")
ax.set_ylabel(feature)
ax.set_title(f"{feature} over time")

plt.xticks(rotation=45)
st.pyplot(fig)
