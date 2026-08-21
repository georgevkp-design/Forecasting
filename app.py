"""
Sales Forecasting Dashboard
----------------------------
Cross-sectional regression dashboard: trains multiple models to predict
target_sales from six financial/macro features, compares them, and lets
the user generate predictions for new ("future") input scenarios.

Note: the source data has no date/time column, so "future" predictions here
mean "predictions for hypothetical/unseen feature values", not a
time-series extrapolation.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 42
TARGET = "target_sales"
REQUIRED_COLS = [
    "sales", "market_indicator_1", "market_indicator_2",
    "gdp_growth", "unemployment_rate", "inflation_rate", TARGET,
]

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes_or_path):
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file_bytes_or_path))
    return pd.read_csv(file_bytes_or_path)


def validate_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------
def build_models(selected, has_xgb):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    catalog = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=SEED))]),
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
        ),
        "Lasso": GridSearchCV(
            Pipeline([("scaler", StandardScaler()),
                      ("model", Lasso(random_state=SEED, max_iter=10000))]),
            param_grid={"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
        ),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ]),
    }
    if has_xgb:
        catalog["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=-1, verbosity=0)),
        ])
    return {name: m for name, m in catalog.items() if name in selected}


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


@st.cache_resource(show_spinner=False)
def train_all(df_hash, df, features, model_names, test_size, has_xgb):
    """Trains models; cached on a hash of the data + config so re-runs are free."""
    X = df[features]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=SEED)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = build_models(model_names, has_xgb)

    rows, fitted, preds = [], {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        cv_rmse = -cross_val_score(
            model, X_tr, y_tr, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        ).mean()
        pred = model.predict(X_te)
        preds[name] = pred
        rows.append({
            "model": name,
            "CV_RMSE": cv_rmse,
            "MAE": mean_absolute_error(y_te, pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
            "R2": r2_score(y_te, pred),
            "MAPE_%": mape(y_te.values, pred),
        })

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return {
        "results": results,
        "fitted": fitted,
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "preds": preds,
        "features": features,
    }


def importance_frame(fitted_model, feature_names):
    est = fitted_model.best_estimator_ if hasattr(fitted_model, "best_estimator_") else fitted_model
    step = est.named_steps["model"]
    if hasattr(step, "feature_importances_"):
        return pd.DataFrame({"feature": feature_names, "value": step.feature_importances_}), "importance"
    return pd.DataFrame({"feature": feature_names, "value": step.coef_}), "coefficient (scaled)"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

uploaded = st.sidebar.file_uploader("Upload training CSV", type=["csv"])
sample_path = "simulated_financial_forecasting_data__1_.csv"

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    df = load_data(raw_bytes)
    data_source = uploaded.name
else:
    try:
        df = load_data(sample_path)
        data_source = f"bundled sample ({sample_path})"
        st.sidebar.info("No file uploaded — using the bundled sample dataset.")
    except FileNotFoundError:
        df = None
        data_source = None
        st.sidebar.warning("Upload a CSV to get started.")

if df is not None:
    missing = validate_columns(df)
    if missing:
        st.error(f"The uploaded CSV is missing required columns: {missing}")
        st.stop()

    all_features = [c for c in REQUIRED_COLS if c != TARGET]

    exclude_sales = st.sidebar.checkbox(
        "Exclude 'sales' feature (leakage check)",
        value=False,
        help="'sales' is almost perfectly correlated with the target. "
             "Toggle this to see how much signal the other features carry alone.",
    )
    features = [f for f in all_features if not (exclude_sales and f == "sales")]

    model_options = ["LinearRegression", "Ridge", "Lasso", "RandomForest"]
    if HAS_XGB:
        model_options.append("XGBoost")
    else:
        st.sidebar.caption("XGBoost not installed — omitted from model list.")

    selected_models = st.sidebar.multiselect(
        "Models to train", model_options, default=model_options
    )

    test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    train_clicked = st.sidebar.button("🚀 Train models", type="primary", use_container_width=True)

    if train_clicked:
        if not selected_models:
            st.sidebar.error("Select at least one model.")
        else:
            with st.spinner("Training models..."):
                df_hash = pd.util.hash_pandas_object(df).sum()
                st.session_state["run"] = train_all(
                    df_hash, df, tuple(features), tuple(selected_models), test_size, HAS_XGB
                )
                st.session_state["run_features"] = features
            st.sidebar.success("Training complete.")


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("📊 Sales Forecasting Dashboard")
st.caption(
    "This dataset has no date column, so this dashboard performs cross-sectional "
    "regression — predicting `target_sales` from feature values — rather than "
    "time-series forecasting. Predictions below are for scenarios you specify, "
    "not for calendar dates."
)

if df is None:
    st.stop()

tab_overview, tab_eda, tab_compare, tab_importance, tab_predict = st.tabs(
    ["Overview", "EDA", "Model Comparison", "Feature Importance", "Predict"]
)

# ---- Overview -------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset overview")
    st.caption(f"Source: {data_source}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))

    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data types**")
        st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
    with col2:
        st.markdown("**Summary statistics**")
        st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("**Correlation matrix**")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ---- EDA --------------------------------------------------------------
with tab_eda:
    st.subheader("Feature distributions")
    cols = list(df.columns)
    n = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(14, 3.2 * ((n + 2) // 3)))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Features vs target")
    feat_cols = [c for c in df.columns if c != TARGET]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), feat_cols):
        ax.scatter(df[col], df[TARGET], s=8, alpha=0.4, color="#4C72B0")
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
        ax.set_title(f"r = {df[col].corr(df[TARGET]):.3f}")
    for ax in axes.ravel()[len(feat_cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---- Model Comparison -------------------------------------------------
with tab_compare:
    st.subheader("Model comparison")
    if "run" not in st.session_state:
        st.info("Configure options in the sidebar and click **Train models**.")
    else:
        run = st.session_state["run"]
        results = run["results"]
        st.dataframe(results.round(4), use_container_width=True)

        best_name = results.loc[0, "model"]
        best_row = results.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best_name)
        c2.metric("RMSE", f"{best_row['RMSE']:.2f}")
        c3.metric("R²", f"{best_row['R2']:.4f}")
        c4.metric("MAPE", f"{best_row['MAPE_%']:.2f}%")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=results, x="model", y="R2", ax=ax, palette="Set2")
        ax.axhline(0, color="black", lw=1)
        ax.set_title("Test R² by model")
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close(fig)

        best_model = run["fitted"][best_name]
        y_te = run["y_te"]
        pred_te = run["preds"][best_name]
        resid = y_te - pred_te

        st.markdown(f"**Diagnostics — {best_name}**")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].scatter(y_te, pred_te, s=12, alpha=0.5, color="#4C72B0")
        lims = [min(y_te.min(), pred_te.min()), max(y_te.max(), pred_te.max())]
        axes[0].plot(lims, lims, "k--", lw=1.5)
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].set_title("Predicted vs actual")

        axes[1].scatter(pred_te, resid, s=12, alpha=0.5, color="#55A868")
        axes[1].axhline(0, color="black", ls="--")
        axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals vs predicted")

        sns.histplot(resid, kde=True, ax=axes[2], color="#8172B2")
        axes[2].axvline(0, color="black", ls="--")
        axes[2].set_title("Residual distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---- Feature Importance -------------------------------------------------
with tab_importance:
    st.subheader("Feature importance / coefficients")
    if "run" not in st.session_state:
        st.info("Train models first from the sidebar.")
    else:
        run = st.session_state["run"]
        model_choice = st.selectbox("Model", list(run["fitted"].keys()))
        imp, kind = importance_frame(run["fitted"][model_choice], run["features"])
        imp = imp.reindex(imp["value"].abs().sort_values().index)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(imp["feature"], imp["value"], color="#4C72B0")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{model_choice} — {kind}")
        st.pyplot(fig)
        plt.close(fig)

# ---- Predict -------------------------------------------------------------
with tab_predict:
    st.subheader("Generate predictions")
    if "run" not in st.session_state:
        st.info("Train models first from the sidebar, then come back here to predict.")
    else:
        run = st.session_state["run"]
        model_choice = st.selectbox(
            "Model to use for prediction", list(run["fitted"].keys()), key="predict_model"
        )
        model = run["fitted"][model_choice]
        features = run["features"]

        st.markdown("#### Single scenario")
        st.caption(
            "Enter feature values for a hypothetical case — e.g. next quarter's "
            "expected conditions — and get a predicted target_sales."
        )
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            lo, hi = float(df[feat].min()), float(df[feat].max())
            default = float(df[feat].mean())
            input_vals[feat] = cols[i % 3].number_input(
                feat, min_value=lo, value=default, step=(hi - lo) / 100 or 1.0
            )

        if st.button("Predict", type="primary"):
            row = pd.DataFrame([input_vals])[features]
            pred = float(model.predict(row)[0])
            st.success(f"Predicted {TARGET}: **{pred:,.2f}**")

        st.divider()
        st.markdown("#### Batch prediction")
        st.caption("Upload a CSV with the same feature columns to score many rows at once.")
        batch_file = st.file_uploader("Upload CSV to score", type=["csv"], key="batch")
        if batch_file is not None:
            new_df = pd.read_csv(batch_file)
            missing_feats = [f for f in features if f not in new_df.columns]
            if missing_feats:
                st.error(f"Uploaded file is missing required columns: {missing_feats}")
            else:
                new_df["predicted_target_sales"] = model.predict(new_df[features])
                st.dataframe(new_df, use_container_width=True)
                csv_bytes = new_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions", data=csv_bytes,
                    file_name="predictions.csv", mime="text/csv"
                )

        st.divider()
        st.markdown("#### Download trained model")
        buf = io.BytesIO()
        joblib.dump({"model": model, "features": features, "name": model_choice}, buf)
        st.download_button(
            "⬇️ Download model (.joblib)", data=buf.getvalue(),
            file_name=f"{model_choice}_target_sales_model.joblib",
        )
