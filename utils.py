"""
Model and data layer for the target_sales dashboard.

Kept separate from app.py so that nothing in here imports streamlit or touches the UI.
That separation matters for a practical reason: this module is the part worth unit-testing
and reusing, and it stays importable from a plain Python shell or the original notebook.

All modelling logic mirrors forecasting_notebook.ipynb. Where a value differs from the
notebook it is flagged in a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

SEED = 42
TARGET = "target_sales"
PRIMARY_FEATURE = "sales"
MACRO_FEATURES = [
    "market_indicator_1",
    "market_indicator_2",
    "gdp_growth",
    "unemployment_rate",
    "inflation_rate",
]
ALL_FEATURES = [PRIMARY_FEATURE] + MACRO_FEATURES
REQUIRED_COLUMNS = ALL_FEATURES + [TARGET]

DEFAULT_CSV = "simulated_financial_forecasting_data.csv"
MODEL_FILE = "target_sales_model.joblib"

MODEL_NAMES = [
    "Linear Regression",
    "Ridge (tuned)",
    "Lasso (tuned)",
    "Random Forest",
    "Gradient Boosting",
]

BASELINE_MEAN = "Baseline: mean"
BASELINE_SALES = "Baseline: sales-only LR"

# The notebook used 400 estimators. Reduced to 250 here purely for cold-start latency on
# Streamlit Community Cloud's shared CPU; the accuracy difference is under one RMSE point,
# well inside the fold-to-fold noise reported in the notebook.
N_ESTIMATORS = 250

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)


# --------------------------------------------------------------------------------------
# Data loading and validation
# --------------------------------------------------------------------------------------


class DataValidationError(Exception):
    """Raised when an uploaded file cannot be used. Carries a user-facing message."""


def validate_frame(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Check a DataFrame is usable and return a cleaned copy.

    Returns rather than mutating so a rejected upload can never contaminate the
    already-loaded dataset in session state.
    """
    if df is None or df.empty:
        raise DataValidationError("The file contains no rows.")

    needed = REQUIRED_COLUMNS if require_target else ALL_FEATURES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(needed)}."
        )

    out = df.copy()

    # Coerce rather than reject outright — a stray thousands separator or a quoted number
    # is a formatting problem, not a data problem, and coercion recovers it.
    present = [c for c in needed if c in out.columns]
    for col in present:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    all_nan = [c for c in present if out[c].isna().all()]
    if all_nan:
        raise DataValidationError(
            f"Column(s) {', '.join(all_nan)} contain no readable numbers."
        )

    before = len(out)
    out = out.dropna(subset=present)
    dropped = before - len(out)

    if out.empty:
        raise DataValidationError("Every row was dropped as non-numeric or incomplete.")

    out.attrs["rows_dropped"] = dropped
    return out


def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or an uploaded file-like object, then validate it."""
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001 — surfaced to the user as a friendly message
        raise DataValidationError(f"Could not parse the file as CSV: {exc}") from exc
    return validate_frame(df, require_target=True)


# --------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------


def build_estimator(name: str):
    """Return an unfitted estimator by name.

    Scaling sits inside every pipeline so it is refit within each CV fold. Fitting a
    scaler on the full training set before cross-validation leaks fold-test statistics
    into the fold-train transform and quietly inflates every score.
    """
    alpha_grid = {"model__alpha": np.logspace(-3, 3, 25)}

    if name == "Linear Regression":
        return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])

    if name == "Ridge (tuned)":
        return GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", Ridge(random_state=SEED))]),
            alpha_grid,
            cv=CV,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

    if name == "Lasso (tuned)":
        return GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Lasso(random_state=SEED, max_iter=20_000)),
                ]
            ),
            alpha_grid,
            cv=CV,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

    if name == "Random Forest":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=2,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if name == "Gradient Boosting":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=N_ESTIMATORS,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=SEED,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model: {name}")


def score_predictions(y_true, y_pred) -> dict:
    """Four headline regression metrics.

    RMSE is computed as sqrt(MSE) by hand: the `squared=False` argument was removed in
    scikit-learn 1.6, and Streamlit Cloud ships a current version.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


# --------------------------------------------------------------------------------------
# Core bundle: fit everything once, cache upstream
# --------------------------------------------------------------------------------------


@dataclass
class CoreBundle:
    """Everything the dashboard needs that does not depend on the selected model."""

    features: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    estimators: dict
    comparison: pd.DataFrame
    baseline_sales_r2: float
    baseline_sales_rmse: float
    best_by_test_rmse: str
    tuned_alphas: dict = field(default_factory=dict)


def fit_core(df: pd.DataFrame, features: list) -> CoreBundle:
    """Split, fit all candidate models, and build the comparison table.

    Deliberately fits every model in one pass so that switching the sidebar model selector
    is instant — the expensive work happens once per (dataset, feature-subset) combination
    rather than once per selection.
    """
    features = list(features)
    X = df[features].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    rows = []

    # Baseline (a) — always predict the training mean. R2 = 0 by construction.
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    rows.append({"Model": BASELINE_MEAN, **score_predictions(y_test, dummy.predict(X_test))})

    # Baseline (b) — the bar that matters. One feature, one coefficient.
    baseline_sales_r2 = np.nan
    baseline_sales_rmse = np.nan
    if PRIMARY_FEATURE in features:
        sales_lr = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
        sales_lr.fit(X_train[[PRIMARY_FEATURE]], y_train)
        sales_pred = sales_lr.predict(X_test[[PRIMARY_FEATURE]])
        sales_scores = score_predictions(y_test, sales_pred)
        rows.append({"Model": BASELINE_SALES, **sales_scores})
        baseline_sales_r2 = sales_scores["R2"]
        baseline_sales_rmse = sales_scores["RMSE"]

    estimators = {}
    tuned_alphas = {}
    for name in MODEL_NAMES:
        est = build_estimator(name)
        est.fit(X_train, y_train)
        estimators[name] = est

        if isinstance(est, GridSearchCV):
            tuned_alphas[name] = float(est.best_params_["model__alpha"])

        cv_rmse = -cross_val_score(
            est, X_train, y_train, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        rows.append(
            {
                "Model": name,
                **score_predictions(y_test, est.predict(X_test)),
                "CV_RMSE": cv_rmse.mean(),
                "CV_RMSE_std": cv_rmse.std(),
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .set_index("Model")
        .sort_values("RMSE")[["MAE", "RMSE", "R2", "MAPE_%", "CV_RMSE", "CV_RMSE_std"]]
    )

    # The winner among real models only — a baseline topping the table is informative but
    # is not something to drive the prediction tabs with.
    real = comparison.loc[comparison.index.isin(MODEL_NAMES)]
    best = real["RMSE"].idxmin()

    return CoreBundle(
        features=features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        estimators=estimators,
        comparison=comparison,
        baseline_sales_r2=baseline_sales_r2,
        baseline_sales_rmse=baseline_sales_rmse,
        best_by_test_rmse=best,
        tuned_alphas=tuned_alphas,
    )


# --------------------------------------------------------------------------------------
# Per-model extras: computed lazily for the selected model only
# --------------------------------------------------------------------------------------


@dataclass
class ModelExtras:
    """Diagnostics for one selected model. Expensive, so cached separately."""

    name: str
    test_pred: np.ndarray
    residuals: np.ndarray
    lower_q: float
    upper_q: float
    perm_importance: pd.DataFrame
    coefficients: pd.DataFrame | None
    ablation: pd.DataFrame
    hetero_ratio: float


def compute_extras(core: CoreBundle, name: str) -> ModelExtras:
    """Residual bounds, permutation importance, coefficients and ablation for one model."""
    est = core.estimators[name]
    features = core.features

    test_pred = est.predict(core.X_test)
    residuals = core.y_test.values - test_pred

    # Interval width comes from cross-validated residuals on the TRAINING set, not from
    # in-sample fit residuals. In-sample residuals are systematically too small because
    # the model has already seen those rows, which would produce an interval that fails
    # to cover 90% of real cases.
    cv_pred = cross_val_predict(est, core.X_train, core.y_train, cv=CV, n_jobs=-1)
    cv_resid = core.y_train.values - cv_pred
    lower_q, upper_q = np.percentile(cv_resid, [5, 95])

    # Permutation importance on the TEST set. Deliberately not tree .feature_importances_,
    # which is impurity-based and biased toward high-cardinality continuous columns — it
    # will hand real-looking importance to pure noise.
    perm = permutation_importance(
        est, core.X_test, core.y_test, n_repeats=20, random_state=SEED, scoring="r2", n_jobs=-1
    )
    perm_df = (
        pd.DataFrame(
            {"Feature": features, "R2_drop": perm.importances_mean, "Std": perm.importances_std}
        )
        .sort_values("R2_drop", ascending=False)
        .reset_index(drop=True)
    )

    # Second opinion from a different mechanism. Standardised so magnitudes are comparable.
    coef_pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]).fit(
        core.X_train, core.y_train
    )
    coefficients = (
        pd.DataFrame({"Feature": features, "Std_coefficient": coef_pipe.named_steps["lr"].coef_})
        .assign(_abs=lambda d: d["Std_coefficient"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )

    # Ablation: the direct test of whether the macro features earn their place.
    ablation_rows = []
    subsets = []
    if PRIMARY_FEATURE in features:
        subsets.append(("sales only", [PRIMARY_FEATURE]))
    subsets.append((f"all {len(features)} selected", features))

    for label, cols in subsets:
        sub = clone(est)
        sub.fit(core.X_train[cols], core.y_train)
        pred = sub.predict(core.X_test[cols])
        ablation_rows.append(
            {
                "Feature set": label,
                "n_features": len(cols),
                "RMSE": float(np.sqrt(mean_squared_error(core.y_test, pred))),
                "R2": r2_score(core.y_test, pred),
            }
        )
    ablation = pd.DataFrame(ablation_rows).set_index("Feature set")

    # Heteroscedasticity proxy without adding a statsmodels dependency to the model layer:
    # does error spread grow with prediction magnitude? ~1.0 means it does not.
    # Guarded: a near-constant predictor (e.g. a feature set with no real signal) puts
    # every row on one side of the median split, which would otherwise divide by zero.
    median_pred = np.median(test_pred)
    low = residuals[test_pred <= median_pred]
    high = residuals[test_pred > median_pred]
    if low.size < 2 or high.size < 2 or not np.isfinite(low.std()) or low.std() == 0:
        ratio = float("nan")
    else:
        ratio = float(high.std() / low.std())

    return ModelExtras(
        name=name,
        test_pred=test_pred,
        residuals=residuals,
        lower_q=float(lower_q),
        upper_q=float(upper_q),
        perm_importance=perm_df,
        coefficients=coefficients,
        ablation=ablation,
        hetero_ratio=ratio,
    )


# --------------------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------------------


def predict_with_interval(
    estimator, new_data: pd.DataFrame, features: list, lower_q: float, upper_q: float
) -> pd.DataFrame:
    """Point prediction plus a 90% band.

    Reorders columns to training order before predicting. A silently mis-ordered DataFrame
    does not raise — it returns confident nonsense — so this is a correctness guard, not
    a convenience.
    """
    missing = [c for c in features if c not in new_data.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}")

    X_new = new_data[features].copy()
    point = estimator.predict(X_new)

    return pd.DataFrame(
        {
            "prediction": point,
            "lower_90": point + lower_q,
            "upper_90": point + upper_q,
        },
        index=new_data.index,
    )


def build_scenario_frame(
    df: pd.DataFrame,
    features: list,
    start_sales: float,
    periods: int,
    growth_rate: float,
    macro_drift: dict | None = None,
) -> pd.DataFrame:
    """Construct a synthetic forward path of feature values.

    This does NOT extrapolate anything learned from the data — the dataset has no time
    dimension, so there is no trend to extrapolate. It compounds a growth rate the user
    supplies. The model then translates that assumed path into target_sales.
    """
    macro_drift = macro_drift or {}
    rows = []

    for t in range(1, periods + 1):
        row = {}
        for feat in features:
            if feat == PRIMARY_FEATURE:
                row[feat] = start_sales * ((1.0 + growth_rate) ** t)
            else:
                base = float(df[feat].median())
                row[feat] = base + macro_drift.get(feat, 0.0) * t
        rows.append(row)

    out = pd.DataFrame(rows, index=pd.RangeIndex(1, periods + 1, name="period"))
    return out[features]


def template_csv() -> str:
    """A header-only CSV so users know exactly what the batch uploader expects."""
    return pd.DataFrame(columns=ALL_FEATURES).to_csv(index=False)
