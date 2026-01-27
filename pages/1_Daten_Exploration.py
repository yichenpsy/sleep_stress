import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.markdown(
    """
    <style>
    .block-container {
        padding: 2.5rem 1rem 2.5rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Data exploration")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/data_final.csv")

df = load_data()


#
df["sleep_sum"] = df["sleep_sum"]/60

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
    ["mean_hrv", "sleep_quality", "sleep_sum", "stepCount", "sportTime"]
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


# "### Choose the variable to plot"
# feature = st.selectbox("Feature",
#                        ["mean_hrv","sleep_quality", "sleep_sum","stepCount", "sportTime"])
#
#
# col1, col2 = st.columns([1,1])
#
# # --- Histogram ---
# with col1:
#     fig1, ax1 = plt.subplots()
#     df[feature].hist(ax=ax1, bins=20)
#     ax1.set_title("Histogram")
#     st.pyplot(fig1)
#
# # --- Boxplot ---
# with col2:
#     fig2, ax2 = plt.subplots()
#     ax2.boxplot(df[feature].dropna())
#     ax2.set_title("Boxplot")
#     st.pyplot(fig2)

