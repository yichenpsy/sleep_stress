"""App-owned interface translations. English text is the translation key."""
import streamlit as st

# Change this to "de" to use German as the default language.
DEFAULT_LANGUAGE = "en"

GERMAN = {
    "Sleep & HRV": "Schlaf & HRV",
    "Overview": "Übersicht",
    "Data Exploration": "Datenexploration",
    "Visualization": "Visualisierung",
    "Ridge Regression": "Ridge-Regression",
    "Logistic Regression": "Logistische Regression",
    "Sleep, Sport & Stress Dashboard": "Dashboard für Schlaf, Sport & Stress",
    "This app analyzes my sleep, HRV, and activity and shows relationships and predictions.": "Diese App analysiert meinen Schlaf, meine HRV und meine Aktivität und zeigt Zusammenhänge und Vorhersagen.",
    "This app analyzes my sleep, HRV, and activity from 2022 to 2025 and shows relationships and predictions.": "Diese App analysiert meinen Schlaf, meine HRV und meine Aktivität von 2022 bis 2025 und zeigt Zusammenhänge und Vorhersagen.",
    "Data source: Apple Watch": "Datenquelle: Apple Watch",
    "Days": "Tage",
    "Average HRV (lower values indicate higher stress)": "Mittlere HRV (niedrigere Werte deuten auf höheren Stress hin)",
    "Average Sleep Quality [0, 100]": "Mittlere Schlafqualität [0, 100]",
    "Average Sleep Duration (h)": "Mittlere Schlafdauer (h)",
    "Average Daily Steps": "Mittlere Schritte pro Tag",
    "Average Daily Sport Time (min)": "Mittlere Sportdauer pro Tag (Min.)",
    "Date": "Datum",
    "Filtered Days": "Gefilterte Tage",
    "Variable": "Variable",
    "Distribution": "Verteilung",
    "Count": "Anzahl",
    "Feature": "Merkmal",
    "{feature} over time": "{feature} im Zeitverlauf",
    "Histogram of {feature}": "Histogramm von {feature}",
    "Boxplot of {feature}": "Boxplot von {feature}",
    "Relationship between {feature} and HRV": "Zusammenhang zwischen {feature} und HRV",
    "{feature} vs HRV": "{feature} im Vergleich zur HRV",
    "Correlation Matrix (Numeric Features)": "Korrelationsmatrix (numerische Merkmale)",
    "Pearson Correlation Matrix": "Pearson-Korrelationsmatrix",
    "HRV Prediction": "HRV-Vorhersage",
    "HRV Prediction with Ridge Regression": "HRV-Vorhersage mit Ridge-Regression",
    "Train Ridge Regression Model": "Ridge-Regressionsmodell trainieren",
    "Train Model": "Modell trainieren",
    "Training model...": "Modell wird trainiert...",
    "Model trained and saved successfully.": "Modell erfolgreich trainiert und gespeichert.",
    "Model Performance (Test Set)": "Modellleistung (Testdaten)",
    "Best α": "Bestes α",
    "Actual mean HRV": "Tatsächliche mittlere HRV",
    "Predicted mean HRV": "Vorhergesagte mittlere HRV",
    "Actual vs Predicted (Test Set)": "Tatsächliche und vorhergesagte Werte (Testdaten)",
    "Predict HRV (User Input)": "HRV vorhersagen (eigene Eingaben)",
    "Sleep duration (hours)": "Schlafdauer (Stunden)",
    "Sleep quality": "Schlafqualität",
    "Step count": "Schrittzahl",
    "Sport time (min)": "Sportdauer (Min.)",
    "Predict HRV": "HRV vorhersagen",
    "Predicted mean HRV: **{value:.2f}**": "Vorhergesagte mittlere HRV: **{value:.2f}**",
    "Please train the model first to enable prediction.": "Bitte zuerst das Modell trainieren, um Vorhersagen zu ermöglichen.",
    "HRV Stress Classification": "HRV-Stressklassifikation",
    "HRV Stress Classification with Logistic Regression": "HRV-Stressklassifikation mit logistischer Regression",
    "Train Logistic Regression Model": "Logistisches Regressionsmodell trainieren",
    "Training logistic regression...": "Logistische Regression wird trainiert...",
    "Accuracy": "Genauigkeit",
    "Positive class: **low HRV (high stress)**": "Positive Klasse: **niedrige HRV (hoher Stress)**",
    "High HRV": "Hohe HRV",
    "Low HRV": "Niedrige HRV",
    "Confusion Matrix – Logistic Regression": "Konfusionsmatrix – Logistische Regression",
    "Predicted label": "Vorhergesagte Klasse",
    "True label": "Tatsächliche Klasse",
    "Show classification report": "Klassifikationsbericht anzeigen",
    "Predict Stress Level (User Input)": "Stressniveau vorhersagen (eigene Eingaben)",
    "Predict Stress Level": "Stressniveau vorhersagen",
    "⚠️ **High stress predicted**  \nProbability (low HRV): **{value:.2f}**": "⚠️ **Hoher Stress vorhergesagt**  \nWahrscheinlichkeit (niedrige HRV): **{value:.2f}**",
    "✅ **Low stress predicted**  \nProbability (low HRV): **{value:.2f}**": "✅ **Niedriger Stress vorhergesagt**  \nWahrscheinlichkeit (niedrige HRV): **{value:.2f}**",
    "Please train the model first.": "Bitte zuerst das Modell trainieren.",
    "Precision": "Präzision",
    "Recall": "Sensitivität",
    "F1 score": "F1-Wert",
    "Support": "Anzahl",
    "Macro average": "Makro-Mittelwert",
    "Weighted average": "Gewichteter Mittelwert",
    "Dataset Overview": "Datensatzübersicht",
    "Statistics": "Statistiken",
    "{rows} rows, {columns} columns": "{rows} Zeilen, {columns} Spalten",
    "Core Sleep (min)": "Kernschlaf (Min.)",
    "Deep Sleep (min)": "Tiefschlaf (Min.)",
    "REM Sleep (min)": "REM-Schlaf (Min.)",
    "Unspecified Sleep (min)": "Nicht zugeordneter Schlaf (Min.)",
    "Awake (min)": "Wachzeit (Min.)",
    "Sleep Duration (min)": "Schlafdauer (Min.)",
    "Sleep Duration (h)": "Schlafdauer (h)",
    "Sleep Efficiency": "Schlafeffizienz",
    "Deep/REM Ratio": "Tiefschlaf-/REM-Verhältnis",
    "Deep/REM Score": "Tiefschlaf-/REM-Wert",
    "Sleep Quality": "Schlafqualität",
    "Mean HRV (ms)": "Mittlere HRV (ms)",
    "Daily Steps": "Schritte pro Tag",
    "Sport Time (min)": "Sportdauer (Min.)",
    """### Model Description

**Target:**
- **mean_hrv** — Mean heart rate variability (HRV) as an indicator of stress and recovery
  *(Higher HRV = lower stress level)*

**Features:**
- **sleep_sum** — Total sleep duration (minutes)
- **sleep_quality** — Normalized sleep quality score (0–100)
- **stepCount** — Daily step count
- **sportTime** — Daily sport duration (minutes)

**Model:**
- Ridge regression (linear regression with L2 regularization)
- Goal: Predict HRV from sleep and activity patterns
""": """### Modellbeschreibung

**Zielvariable (Target):**
- **mean_hrv** — Durchschnittliche Herzratenvariabilität (HRV) als Indikator für Stress und Erholung
  *(Höhere HRV = geringeres Stressniveau)*

**Prädiktoren (Features):**
- **sleep_sum** — Gesamte Schlafdauer (in Minuten)
- **sleep_quality** — Normalisierter Schlafqualitäts-Score (0–100)
- **stepCount** — Anzahl der täglichen Schritte
- **sportTime** — Sportdauer pro Tag (in Minuten)

**Modell:**
- Ridge Regression (lineares Regressionsmodell mit L2-Regularisierung)
- Ziel: Vorhersage der HRV basierend auf Schlaf- und Aktivitätsverhalten
""",
    """### Model Description

**Binary target:**
- `1` → low HRV (**high stress level**)
- `0` → high HRV (**low stress level**)

The target is defined using quantiles:
- Bottom 25% → **low_hrv = 1**
- Top 25% → **low_hrv = 0**
- The middle 50% are excluded

**Features:**
- **sleep_sum** — Total sleep duration (minutes)
- **sleep_quality** — Sleep quality score (0–100)
- **stepCount** — Daily step count
- **sportTime** — Sport duration (minutes)

**Model:**
- Logistic regression (binary classification)
- Standardized features
- Evaluation using accuracy and ROC-AUC
""": """### Modellbeschreibung

**Zielvariable (binär):**
- `1` → niedrige HRV (**hohes Stressniveau**)
- `0` → hohe HRV (**niedriges Stressniveau**)

Die Zielvariable wird über Quantile gebildet:
- untere 25 % → **low_hrv = 1**
- obere 25 % → **low_hrv = 0**
- mittlere 50 % werden ausgeschlossen

**Prädiktoren:**
- **sleep_sum** — Gesamte Schlafdauer (Minuten)
- **sleep_quality** — Schlafqualität-Score (0–100)
- **stepCount** — Anzahl Schritte pro Tag
- **sportTime** — Sportdauer (Minuten)

**Modell:**
- Logistische Regression (binäre Klassifikation)
- Standardisierung der Features
- Evaluation mit Genauigkeit und ROC-AUC
""",
}

COLUMN_LABELS = {
    "date": "Date",
    "sleepCore": "Core Sleep (min)",
    "sleepDeep": "Deep Sleep (min)",
    "sleepREM": "REM Sleep (min)",
    "sleepUnspecified": "Unspecified Sleep (min)",
    "awake": "Awake (min)",
    "sleep_sum": "Sleep Duration (min)",
    "sleep_sum_hours": "Sleep Duration (h)",
    "sleep_efficiency": "Sleep Efficiency",
    "deep_rem_ratio": "Deep/REM Ratio",
    "deep_rem_score": "Deep/REM Score",
    "sleep_quality": "Sleep Quality",
    "mean_hrv": "Mean HRV (ms)",
    "stepCount": "Daily Steps",
    "sportTime": "Sport Time (min)",
}


def t(english, **values):
    """Translate visible text, then fill named placeholders such as {feature}."""
    text = GERMAN.get(english, english) if st.session_state.get("language", DEFAULT_LANGUAGE) == "de" else english
    return text.format(**values) if values else text


def column_label(column):
    return t(COLUMN_LABELS.get(column, column))


def display_data(frame):
    """Rename a display copy only; the original model input columns stay intact."""
    return frame.rename(columns=column_label)
