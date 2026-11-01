from utils.i18n import t, column_label, display_data
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title(t('Visualization'))

@st.cache_data
def load_data(version="v2"):
    return pd.read_csv("data/processed/data_final.csv")

df = load_data(version="2026-01-27")
df["date"] = pd.to_datetime(df["date"])

st.subheader(t("Distribution"))
feature = st.selectbox(
    t('Variable'),
    ["mean_hrv", "sleep_quality", "sleep_sum_hours", "stepCount", "sportTime"],
    format_func={name: column_label(name) for name in df.columns}.__getitem__, key="2_Visualisierung_variable"
)

# =========================
# Row 1: Histogram + Boxplot
# =========================
col1, col2 = st.columns([1, 1])

# --- Histogram ---
with col1:
    fig1, ax1 = plt.subplots()
    ax1.hist(df[feature].dropna(), bins=20)
    ax1.set_title(t('Histogram of {feature}', feature=column_label(feature)))
    ax1.set_xlabel(column_label(feature))
    ax1.set_ylabel(t('Count'))
    st.pyplot(fig1)

# --- Boxplot ---
with col2:
    fig2, ax2 = plt.subplots()
    ax2.boxplot(df[feature].dropna())
    ax2.set_title(t('Boxplot of {feature}', feature=column_label(feature)))
    ax2.set_ylabel(column_label(feature))
    st.pyplot(fig2)


# =========================
# Row 2: Scatter + Regression line
# =========================
st.markdown("---")

feature = st.selectbox(
    t('Feature'),
    ["sleep_quality", "sleep_sum_hours", "stepCount", "sportTime"],
    format_func={name: column_label(name) for name in df.columns}.__getitem__, key="scatter_feature"
)

st.subheader(t("Relationship between {feature} and HRV", feature=column_label(feature)))

# Prepare clean data (drop NaNs in both)
df_scatter = df[[feature, "mean_hrv"]].dropna()

x = df_scatter[feature].values
y = df_scatter["mean_hrv"].values

fig3, ax3 = plt.subplots()

# Scatter
ax3.scatter(x, y, alpha=0.6)

# Regression line (simple linear fit)
if len(x) > 1:
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    ax3.plot(x_line, y_line)

ax3.set_title(t('{feature} vs HRV', feature=column_label(feature)))
ax3.set_xlabel(column_label(feature))
ax3.set_ylabel(column_label("mean_hrv"))

st.pyplot(fig3)


# =========================
# Row 3: Correlation Matrix
# =========================
st.markdown("---")
st.subheader(t('Correlation Matrix (Numeric Features)'))

corr_features = ["mean_hrv", "sleep_quality", "sleep_sum_hours", "stepCount", "sportTime"]
corr_matrix = df[corr_features].corr()

fig4, ax4 = plt.subplots()
im = ax4.imshow(corr_matrix, cmap="coolwarm",vmin=-1,vmax=1,aspect="auto")

# Ticks & labels
ax4.set_xticks(range(len(corr_features)))
ax4.set_yticks(range(len(corr_features)))
ax4.set_xticklabels([column_label(c) for c in corr_features], rotation=45, ha="right")
ax4.set_yticklabels([column_label(c) for c in corr_features])

# Add correlation values as text
for i in range(len(corr_features)):
    for j in range(len(corr_features)):
        ax4.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=9)

ax4.set_title(t('Pearson Correlation Matrix'))

fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

st.pyplot(fig4)
