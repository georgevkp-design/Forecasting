# target_sales — prediction dashboard

An interactive Streamlit dashboard built from `forecasting_notebook.ipynb`. Six tabs:
overview, exploratory analysis, model performance, what-if prediction, batch scoring, and
assumption-driven scenario projection.

## What this app is (and is not)

It predicts `target_sales` for a row whose feature values you already know. It is **not a
time-series forecast** — the dataset has no date, time or period column, so there is no
trend to extrapolate. The scenario tab projects forward from a growth rate *you* supply and
is labelled accordingly.

The honest headline: `sales` correlates with `target_sales` at r ≈ 0.985, and the five
macro indicators add **+0.028% relative R²**. This is a one-variable model. The UI is built
to show that rather than hide it — deselect the macro features in the sidebar and watch the
metrics not move.

## Files

```
app.py                                     # UI layer (Streamlit only)
utils.py                                   # data loading, validation, modelling
requirements.txt
.streamlit/config.toml                     # dark theme
simulated_financial_forecasting_data.csv   # bundled so a cold deploy works with no input
```

`target_sales_model.joblib` is optional. If you drop one in the repo root the app detects
and reports it; otherwise the model is trained in-app on first run and cached.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens on <http://localhost:8501>.

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repo** and push these files. The CSV must be committed — without it
   the app starts with no data and waits for an upload.
   ```bash
   git init
   git add app.py utils.py requirements.txt .streamlit/config.toml simulated_financial_forecasting_data.csv README.md
   git commit -m "target_sales dashboard"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app** → **Deploy a public app from GitHub**.
4. Set:
   - **Repository:** `<you>/<repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**. First build takes 2–4 minutes while dependencies install.

**Expected cold start:** roughly 20–40 seconds on the first load while all five models fit.
Everything is cached afterwards, so tab switching and slider drags are instant. A spinner
covers the initial fit.

### If the build fails

- **`ModuleNotFoundError`** — the import is missing from `requirements.txt`.
- **Blank charts / trendline missing** — `statsmodels` is required by
  `px.scatter(trendline="ols")`. It is in `requirements.txt`; don't strip it.
- **App exceeds resource limits** — lower `N_ESTIMATORS` in `utils.py` (currently 250).

## Notes

- Requires **Python 3.9+**. Community Cloud defaults to a recent 3.x, which is fine.
- RMSE is computed as `np.sqrt(mean_squared_error(...))`. The `squared=False` argument was
  removed in scikit-learn 1.6 and will crash on current Cloud images.
- The app uses a small compatibility shim for `use_container_width` / `width="stretch"`, so
  it runs on both older and current Streamlit releases.

## .gitignore

```
.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
```
