# Sales Forecasting Dashboard

A Streamlit dashboard that trains and compares regression models on
`simulated_financial_forecasting_data__1_.csv` and lets you generate
predictions for new feature scenarios.

**Note:** the dataset has no date/time column, so this is cross-sectional
regression, not time-series forecasting. "Prediction" here means: given a
hypothetical set of feature values (a scenario), predict `target_sales`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).
Place `simulated_financial_forecasting_data__1_.csv` in the same folder as
`app.py` to have it load automatically, or just upload a CSV in the sidebar.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and (optionally) the sample CSV to a
   GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, and click
   "New app".
3. Select the repo, branch, and set the main file path to `app.py`.
4. Click "Deploy". The first build takes a couple of minutes while
   dependencies install.
5. Every push to the connected branch redeploys automatically.

## What's in the dashboard

- **Overview** — shape, dtypes, summary stats, correlation heatmap
- **EDA** — feature distributions, feature-vs-target scatter plots
- **Model Comparison** — trains LinearRegression, Ridge, Lasso,
  RandomForest, and XGBoost; compares MAE/RMSE/R²/MAPE; shows
  predicted-vs-actual and residual plots for the best model
- **Feature Importance** — coefficients or importances per model
- **Predict** — single-scenario prediction form, batch CSV scoring with
  a download button, and a downloadable trained model (`.joblib`)

Use the sidebar to upload data, toggle whether the (dominant) `sales`
feature is included, pick which models to train, and set the test split.
