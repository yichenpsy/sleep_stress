# Sleep, Sport & Stress Dashboard

A Streamlit dashboard for exploring sleep, heart rate variability (HRV), and
activity. The app supports English and German, with interactive charts and two
machine learning pages.

## Features

| Page | What it does |
| --- | --- |
| Overview | Shows daily records and average sleep, HRV, steps, and sport metrics |
| Data Exploration | Filters dates and plots a selected variable over time |
| Visualization | Shows histograms, boxplots, relationships with HRV, and correlations |
| Ridge Regression | Predicts mean HRV and reports R² and mean absolute error |
| Logistic Regression | Classifies low versus high HRV and reports accuracy, ROC-AUC, and a confusion matrix |

On either model page, click **Train Model** before entering values and making a
prediction. Models are stored in the current Streamlit session, rather than saved
as model files. A new session or server restart requires training again.

## Run locally

Run all commands from the project root: the folder containing `app.py` and
`requirements.txt`. In PyCharm, open the project's **Terminal** tab.

If this checkout already has a `.venv`, start the app with:

```bash
.venv/bin/python -m streamlit run app.py
```

For a fresh checkout on macOS or Linux, create the environment and install dependencies first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

The existing local environment runs Python 3.13.3; this is a verified environment,
not a declaration that other Python versions are unsupported. Dependencies include
Streamlit 1.52.2 or later, pandas, NumPy, scikit-learn, and Matplotlib.

- `.venv` is the chosen project-local virtual environment directory.
- `.venv/bin/python` uses its Python interpreter without activating the environment.
- `app.py` is the app entry point; do not run it with plain `python app.py`.
- `requirements.txt` lists the app's Python dependencies.

Open [http://localhost:8501](http://localhost:8501) and keep the terminal running.
`localhost` means your own computer; `8501` is Streamlit's default port. Stop the
server with **Control + C**. If the port is busy, choose another port:

```bash
.venv/bin/python -m streamlit run app.py --server.port 8502
```

`8502` is an example available port; open `http://localhost:8502` when using it.
If a dependency is missing, rerun the `pip install -r requirements.txt` command above.

## Languages

Use **Language / Sprache** in the sidebar to select **English** or **Deutsch**.
Navigation, descriptions, controls, table column labels, and chart labels change
language. The choice persists across pages within the current session. Changing
language retains the current page's prediction inputs and trained models.
Streamlit's built-in toolbar and table menus are separate from app translations.

Translations live in [`utils/i18n.py`](utils/i18n.py):

| Setting or helper | Purpose and what you can change |
| --- | --- |
| `DEFAULT_LANGUAGE = "en"` | Default for a new session; change to `"de"` for German |
| `GERMAN` | English text as the key, German text as its translation |
| `COLUMN_LABELS` | Original CSV column name mapped to a readable English label, which is then translated |
| `t(...)` | Selects the current translation and fills named placeholders |

For example, `{feature}` is replaced with the selected variable's display label,
and `{value:.2f}` displays a number with two decimal places. Keep matching
placeholder names in both languages. Only table display copies are renamed;
original data columns and model calculations retain their original names.

## Data and models

The app reads [`data/processed/data_final.csv`](data/processed/data_final.csv).
It does not execute the preparation notebooks at startup. The supplied table has
752 daily records and 15 columns; those counts may change when the data is updated.

Both models use the following input columns:

| Column | Meaning / unit |
| --- | --- |
| `sleep_sum` | Total sleep duration, minutes |
| `sleep_quality` | Sleep quality score, 0–100 |
| `stepCount` | Daily step count |
| `sportTime` | Daily sport duration, minutes |

The Ridge target is `mean_hrv` in milliseconds. Logistic regression uses the
bottom and top HRV quartiles: `1` means low HRV, `0` means high HRV; the middle
range is excluded. Thresholds are calculated from the loaded data. Prediction
forms accept sleep in hours and convert it to minutes before passing it to a model.

The current training settings are defined in the corresponding files in `views/`:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `test_size` | `0.2` | Reserve 20% of samples for evaluation |
| `random_state` | `42` | Arbitrary fixed seed for a repeatable train/test split |
| Ridge `alphas` | `[0.01, 0.1, 1, 10, 100]` | Regularization strengths considered by `RidgeCV` |
| Logistic `max_iter` | `1000` | Maximum fitting iterations |

Both pipelines standardize their input features. These parameter values can be
changed in the model pages; changing them requires retraining to update results.

## Project structure

```text
app.py                          Entry point, language selector, navigation
utils/dashboard.py              Overview page
utils/i18n.py                   Translations and display labels
views/                          Four exploration and model pages
data/processed/data_final.csv   Prepared input data
notebooks/                      Data preparation, exploration, and model notebooks
tests/test_bilingual_app.py      Bilingual integration check
requirements.txt                App dependencies
```

`app.py` uses `st.navigation` so navigation titles can be translated. Page files
live in `views/` to avoid Streamlit's automatic `pages/` navigation. Existing
feature-page URL paths are retained even when their visible titles change.
`stream_lit_test.py` and `utils/_3_ML_Linear_Regression.py` are standalone scripts,
not pages registered by `app.py`.

## Git remotes and deployment

This local checkout has two configured remotes:

| Local alias | Repository |
| --- | --- |
| `origin_github` | [GitHub: yichenpsy/sleep_stress](https://github.com/yichenpsy/sleep_stress) |
| `origin` | [GitLab: yizh8704/dp_sleep_stress](https://gitlab.bht-berlin.de/yizh8704/dp_sleep_stress) |

These aliases are local Git settings; a fresh clone may use different names.
The current local `main` tracks `origin_github/main`. Use `git remote -v` and
`git branch -vv` to inspect your own checkout. A remote named `origin` is not
necessarily the current branch's default destination.

For **Streamlit Community Cloud**, the expected configuration is:

| Field | Value | Source |
| --- | --- | --- |
| Repository | `yichenpsy/sleep_stress` | GitHub repository above |
| Branch | `main` | Project's main branch |
| Main file path | `app.py` | Application entry point |

The actual hosted app's repository, branch, and URL have not been verified from
the deployment account. Find the app in the [Streamlit console](https://share.streamlit.io/)
and confirm these settings. Record the verified app URL here when available.

When Community Cloud is connected to that GitHub branch, pushing committed
changes to it updates the hosted app automatically. Saving files or committing
locally does not publish them. Pushing only to GitLab does not update a deployment
connected to GitHub. See [Streamlit's app update documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/edit-your-app).

Before pushing, commit the intended changes and integrate any remote commits.
A commit records selected changes locally; a push uploads those commits. A push
alone does not include uncommitted files.

To refresh remote tracking information without merging changes into your files:

```bash
git fetch origin_github
git branch -vv
```

`origin_github` is the local alias listed above. In the branch output, `ahead`
means local commits are missing remotely; `behind` means remote commits are
missing locally. If both appear, integrate the histories using merge or rebase
and resolve any conflicts before pushing.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests
```

The integration check visits all five pages in both languages, trains and uses
both models, and checks that changing languages retains inputs and model state.
It also checks that each model retains its own evaluation data.
