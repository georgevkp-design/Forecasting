"""
target_sales dashboard — Streamlit UI layer.

Companion to forecasting_notebook.ipynb. All modelling lives in utils.py; this file is
presentation only.

Design note that runs through the whole app: the underlying model is a one-variable model.
`sales` explains ~97% of `target_sales` and the five macro features contribute effectively
nothing. The UI is built to make that visible rather than to flatter the result — every R2
appears beside the sales-only baseline, and the scenario tab is labelled a projection
rather than a forecast because the dataset has no time dimension.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils as U

# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title="target_sales — prediction dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Spec asked for use_container_width=True. That argument is past its Streamlit removal
# date, and Community Cloud tracks the latest release, so hardcoding it is a live
# deployment risk. This shim preserves the intent (full-width elements) on both the old
# and new API by selecting the correct keyword at import time.
try:
    from packaging.version import Version as _V

    STRETCH = (
        {"width": "stretch"} if _V(st.__version__) >= _V("1.49") else {"use_container_width": True}
    )
except Exception:  # noqa: BLE001
    STRETCH = {"use_container_width": True}

ACCENT = "#00C2A8"
WARN = "#E8833A"
MUTED = "#8B93A7"
PLOT_TEMPLATE = "plotly_dark"


# ======================================================================================
# Cached data + model layer
# ======================================================================================


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    if not os.path.exists(U.DEFAULT_CSV):
        raise U.DataValidationError(
            f"Bundled dataset '{U.DEFAULT_CSV}' not found. Upload a CSV in the sidebar."
        )
    return U.load_csv(U.DEFAULT_CSV)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, _name: str) -> pd.DataFrame:
    import io

    return U.load_csv(io.BytesIO(file_bytes))


# _df is underscore-prefixed so Streamlit skips hashing the frame itself; data_key carries
# the identity instead. Hashing an 1,000-row DataFrame on every rerun is pure overhead.
@st.cache_resource(show_spinner=False)
def get_core(_df: pd.DataFrame, data_key: str, features: tuple) -> U.CoreBundle:
    return U.fit_core(_df, list(features))


@st.cache_resource(show_spinner=False)
def get_extras(_core: U.CoreBundle, data_key: str, features: tuple, model_name: str):
    return U.compute_extras(_core, model_name)


@st.cache_resource(show_spinner=False)
def try_load_joblib(path: str):
    """Prefer a shipped artefact when one exists. Returns None if absent or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:  # noqa: BLE001 — a bad artefact should degrade, not crash the app
        return None


def frame_key(df: pd.DataFrame) -> str:
    """Cheap stable identity for a DataFrame, used as an explicit cache key."""
    return f"{len(df)}-{int(pd.util.hash_pandas_object(df, index=True).sum()) & 0xFFFFFFFF}"


def styled(df: pd.DataFrame, highlight_index=None, precision: int = 4):
    """Format a metrics table and optionally highlight one row."""
    sty = df.style.format(precision=precision, thousands=",")
    if highlight_index is not None and highlight_index in df.index:

        def _hl(row):
            colour = f"background-color: rgba(0,194,168,0.18); font-weight:600;"
            return [colour if row.name == highlight_index else "" for _ in row]

        sty = sty.apply(_hl, axis=1)
    return sty


def rgba(hex_colour: str, alpha: float) -> str:
    """Translucent fill colour. Plotly's fillcolor needs rgba(); opacity on a filled
    Scatter dims the line too, which is not what we want for a fan chart."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def fig_layout(fig, height=380, title=None):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=45 if title else 20, b=10),
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ======================================================================================
# Sidebar
# ======================================================================================

st.sidebar.title("Controls")

st.sidebar.subheader("Data source")
uploaded = st.sidebar.file_uploader(
    "Upload a CSV (optional)",
    type=["csv"],
    help=f"Must contain: {', '.join(U.REQUIRED_COLUMNS)}",
)

df = None
source_label = ""
try:
    if uploaded is not None:
        df = load_uploaded_data(uploaded.getvalue(), uploaded.name)
        source_label = f"Uploaded — {uploaded.name}"
    else:
        df = load_default_data()
        source_label = f"Bundled — {U.DEFAULT_CSV}"
except U.DataValidationError as exc:
    st.sidebar.error(str(exc))
    st.error(
        "Could not load a usable dataset. Fix the upload or remove it to fall back to the "
        "bundled file."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Unexpected error while loading data: {exc}")
    st.stop()

dropped = df.attrs.get("rows_dropped", 0)
st.sidebar.caption(f"{source_label}  \n**{len(df):,} rows** loaded")
if dropped:
    st.sidebar.warning(f"{dropped} row(s) dropped as non-numeric or incomplete.")

st.sidebar.divider()
st.sidebar.subheader("Model")

selected_model = st.sidebar.selectbox(
    "Active model",
    U.MODEL_NAMES,
    index=U.MODEL_NAMES.index("Lasso (tuned)"),
    help="Drives the diagnostics, prediction and projection tabs. All models are fitted "
    "once and cached, so switching is instant.",
    key="sidebar_model",
)

selected_features = st.sidebar.multiselect(
    "Features used",
    U.ALL_FEATURES,
    default=U.ALL_FEATURES,
    help="Deselect the five macro indicators and watch the metrics barely move. That is "
    "the point of this control.",
    key="sidebar_features",
)

if not selected_features:
    st.sidebar.error("Select at least one feature.")
    st.error("No features selected — nothing to model.")
    st.stop()

data_key = frame_key(df)
features_key = tuple(selected_features)

artefact = try_load_joblib(U.MODEL_FILE)

with st.spinner("Fitting models (first run only — results are cached)…"):
    try:
        core = get_core(df, data_key, features_key)
        extras = get_extras(core, data_key, features_key, selected_model)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model fitting failed: {exc}")
        st.stop()

estimator = core.estimators[selected_model]

if artefact is not None and features_key == tuple(U.ALL_FEATURES):
    st.sidebar.success(f"Loaded artefact `{U.MODEL_FILE}`")
    st.sidebar.caption(
        "A shipped .joblib was found. Metrics below are still recomputed in-app so they "
        "match the current data and feature selection."
    )
else:
    st.sidebar.info("Model trained in-app and cached")

if selected_model in core.tuned_alphas:
    st.sidebar.caption(f"Tuned alpha: `{core.tuned_alphas[selected_model]:.4g}`")

st.sidebar.divider()
with st.sidebar.expander("About & caveats", expanded=False):
    st.markdown(
        """
**This is not a forecast.** The dataset has no date, time or period column. The model
answers *"given this row's inputs, what is its `target_sales`?"* — not *"what happens next
quarter?"*

**Known limitations**

- **Simulated data.** Symmetric distributions, zero missingness, no outliers. Real
  financial data looks nothing like this, so performance here predicts nothing about
  production performance.
- **Correlation, not causation.** Nothing establishes that `sales` *causes* `target_sales`.
  Given r ≈ 0.985 and the naming, the likelier explanation is that `target_sales` was
  constructed from `sales` — in which case the model is recovering an arithmetic identity,
  not learning economics.
- **Small test set.** 200 rows. Differences of a few RMSE points between the top models
  are inside fold-to-fold noise; treat them as tied.
- **Selection-on-test-set bias.** Six models are compared on the same held-out rows, so the
  winner's test metric is mildly optimistic. The CV column is the safer estimate.
- **Interval caveats.** The 90% band is constant-width and covers residual scatter only,
  excluding parameter uncertainty.
        """
    )

# Convenience handles used across tabs.
comparison = core.comparison
best_r2 = comparison.loc[selected_model, "R2"]
best_rmse = comparison.loc[selected_model, "RMSE"]
base_r2 = core.baseline_sales_r2
has_sales = U.PRIMARY_FEATURE in selected_features


# ======================================================================================
# Header
# ======================================================================================

st.title("target_sales — prediction dashboard")
st.caption(
    "Cross-sectional regression on simulated financial data · companion to "
    "`forecasting_notebook.ipynb`"
)

tab_overview, tab_eda, tab_perf, tab_single, tab_batch, tab_scenario = st.tabs(
    [
        "Overview",
        "Exploratory analysis",
        "Model performance",
        "What-if prediction",
        "Batch prediction",
        "Scenario projection",
    ]
)


# ======================================================================================
# Tab 1 — Overview
# ======================================================================================

with tab_overview:
    st.info(
        "This is a **conditional prediction model, not a time-series forecast** — it "
        "estimates `target_sales` for a row whose feature values you already know.",
        icon="ℹ️",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features in use", f"{len(selected_features)} of {len(U.ALL_FEATURES)}")

    delta_r2 = (best_r2 - base_r2) if not np.isnan(base_r2) else None
    c3.metric(
        f"Test R² — {selected_model}",
        f"{best_r2:.4f}",
        delta=f"{delta_r2:+.4f} vs sales-only" if delta_r2 is not None else None,
        help="The delta is the entire value added by modelling beyond a one-variable "
        "linear regression on `sales`.",
    )
    c4.metric("Test RMSE", f"{best_rmse:,.2f}")
    c5.metric(
        "Sales-only R²",
        f"{base_r2:.4f}" if not np.isnan(base_r2) else "n/a",
        help="The bar every model must clear.",
    )

    if delta_r2 is not None and abs(delta_r2) < 0.005:
        st.warning(
            f"**Read the delta.** The active model beats a single-variable regression on "
            f"`sales` by {delta_r2:+.4f} R². The headline {best_r2:.4f} is almost entirely "
            f"one feature's doing, not the model's.",
            icon="⚠️",
        )

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Data preview")
        st.dataframe(df.head(25), **STRETCH, height=340)

    with right:
        st.subheader("Summary statistics")
        st.dataframe(
            df.describe().T.style.format(precision=2, thousands=","),
            **STRETCH,
            height=340,
        )


# ======================================================================================
# Tab 2 — Exploratory analysis
# ======================================================================================

with tab_eda:
    st.subheader("Distributions")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        dist_feature = st.selectbox(
            "Column", U.REQUIRED_COLUMNS, index=0, key="eda_dist_feature"
        )
        marginal = st.radio(
            "Marginal", ["box", "violin", "rug"], horizontal=False, key="eda_marginal"
        )

    with col_b:
        fig = px.histogram(
            df,
            x=dist_feature,
            nbins=45,
            marginal=marginal,
            color_discrete_sequence=[ACCENT],
        )
        st.plotly_chart(
            fig_layout(fig, 400, f"Distribution of {dist_feature}"),
            **STRETCH,
        )

    st.caption(
        "Everything here is broadly symmetric and unimodal — no transform needed, and a "
        "tell that the data is simulated. Real revenue distributions are almost always "
        "right-skewed."
    )

    st.divider()
    st.subheader("Where the signal is")

    c1, c2 = st.columns(2)

    with c1:
        corr = df[U.REQUIRED_COLUMNS].corr()
        fig = px.imshow(
            corr,
            text_auto=".3f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        st.plotly_chart(fig_layout(fig, 460, "Correlation matrix"), **STRETCH)

    with c2:
        target_corr = (
            df[U.REQUIRED_COLUMNS]
            .corr()[U.TARGET]
            .drop(U.TARGET)
            .sort_values()
            .rename("correlation")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        fig = px.bar(
            target_corr,
            x="correlation",
            y="feature",
            orientation="h",
            color="correlation",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        st.plotly_chart(
            fig_layout(fig, 460, f"Correlation with {U.TARGET}"), **STRETCH
        )

    st.error(
        "**One signal, five passengers.** `sales` correlates with the target at ~0.985. "
        "Every macro indicator sits under |0.04| — the range you get from pure noise at "
        "n=1,000.",
        icon="🔍",
    )

    st.divider()
    st.subheader("Relationship to the target")

    scat_feature = st.selectbox(
        "Feature on the x-axis",
        U.ALL_FEATURES,
        index=0,
        key="eda_scatter_feature",
    )
    try:
        fig = px.scatter(
            df,
            x=scat_feature,
            y=U.TARGET,
            trendline="ols",
            opacity=0.45,
            color_discrete_sequence=[ACCENT],
            trendline_color_override=WARN,
        )
    except Exception:  # noqa: BLE001 — statsmodels missing shouldn't kill the tab
        fig = px.scatter(
            df, x=scat_feature, y=U.TARGET, opacity=0.45, color_discrete_sequence=[ACCENT]
        )
        st.caption("Trendline unavailable — install `statsmodels` to enable OLS fitting.")

    r_val = df[scat_feature].corr(df[U.TARGET])
    st.plotly_chart(
        fig_layout(fig, 460, f"{scat_feature} vs {U.TARGET}  (r = {r_val:.4f})"),
        **STRETCH,
    )


# ======================================================================================
# Tab 3 — Model performance
# ======================================================================================

with tab_perf:
    st.subheader("Model comparison")
    st.caption(
        "Sorted by test RMSE, with both baselines included. `CV_RMSE` is the more "
        "trustworthy column — it averages five splits instead of betting on one 200-row "
        "sample."
    )

    st.dataframe(
        styled(comparison, highlight_index=selected_model),
        **STRETCH,
    )

    # Objection noted per spec: ranking on the test set turns it into a selection set.
    st.caption(
        "⚠️ Methodological note: choosing a winner by test RMSE makes the test set a "
        "selection set, so the top row's metric is mildly optimistic. The rigorous "
        "procedure is to select on CV_RMSE and touch the test set once. Both orderings "
        "agree on this data, so nothing downstream changes."
    )

    plot_df = comparison.reset_index()
    fig = px.bar(
        plot_df,
        x="Model",
        y="RMSE",
        color="Model",
        text=plot_df["RMSE"].map(lambda v: f"{v:,.1f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(
        fig_layout(fig, 400, "Test RMSE by model (lower is better)"), **STRETCH
    )

    st.divider()
    st.subheader(f"Diagnostics — {selected_model}")

    y_test = core.y_test.values
    pred = extras.test_pred
    resid = extras.residuals

    d1, d2 = st.columns(2)

    with d1:
        lo, hi = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=pred,
                mode="markers",
                marker=dict(size=6, opacity=0.5, color=ACCENT),
                name="predictions",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(dash="dash", color=WARN, width=2),
                name="perfect prediction",
            )
        )
        fig.update_xaxes(title="Actual")
        fig.update_yaxes(title="Predicted")
        st.plotly_chart(fig_layout(fig, 400, "Predicted vs actual"), **STRETCH)

    with d2:
        fig = px.histogram(resid, nbins=30, color_discrete_sequence=[ACCENT])
        fig.add_vline(x=0, line_dash="dash", line_color=WARN)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="Residual (actual − predicted)")
        st.plotly_chart(
            fig_layout(fig, 400, "Residual distribution"), **STRETCH
        )

    fig = px.scatter(
        x=pred, y=resid, opacity=0.5, color_discrete_sequence=[ACCENT],
        labels={"x": "Predicted", "y": "Residual"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color=WARN)
    st.plotly_chart(fig_layout(fig, 340, "Residuals vs predicted"), **STRETCH)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Residual mean", f"{resid.mean():,.2f}", help="Want ~0. Non-zero = bias.")
    m2.metric("Residual std", f"{resid.std():,.2f}")
    m3.metric("Residual skew", f"{pd.Series(resid).skew():.3f}", help="Want ~0.")
    m4.metric(
        "Heteroscedasticity ratio",
        f"{extras.hetero_ratio:.3f}" if np.isfinite(extras.hetero_ratio) else "n/a",
        help="Error spread in the upper half of predictions divided by the lower half. "
        "Near 1.0 = homoscedastic. Outside 0.67–1.5 invalidates the constant-width "
        "prediction interval used elsewhere in this app.",
    )

    if np.isfinite(extras.hetero_ratio) and not (0.67 <= extras.hetero_ratio <= 1.5):
        st.warning(
            "Residual spread depends on prediction magnitude. The constant-width 90% "
            "interval used in the prediction tabs is too narrow at one end and too wide at "
            "the other. Quantile regression or a conformal method would be more "
            "appropriate.",
            icon="⚠️",
        )

    st.divider()
    st.subheader("Feature importance")
    st.caption(
        "Permutation importance on the **test** set, 20 repeats — deliberately not tree "
        "`.feature_importances_`, which is impurity-biased and hands real-looking "
        "importance to noise columns."
    )

    i1, i2 = st.columns([3, 2])

    with i1:
        pi = extras.perm_importance.sort_values("R2_drop")
        fig = go.Figure(
            go.Bar(
                x=pi["R2_drop"],
                y=pi["Feature"],
                orientation="h",
                error_x=dict(type="data", array=pi["Std"], color=MUTED),
                marker_color=ACCENT,
            )
        )
        fig.add_vline(x=0, line_color=MUTED)
        fig.update_xaxes(title="Drop in test R² when shuffled")
        st.plotly_chart(
            fig_layout(fig, 380, "Permutation importance"), **STRETCH
        )

    with i2:
        st.dataframe(
            extras.coefficients.style.format({"Std_coefficient": "{:,.4f}"}),
            **STRETCH,
            height=250,
        )
        st.caption("Standardised linear coefficients — a second opinion from a different mechanism.")

    signal = extras.perm_importance[
        extras.perm_importance["R2_drop"] > 2 * extras.perm_importance["Std"].clip(lower=1e-12)
    ]["Feature"].tolist()
    noise = [f for f in selected_features if f not in signal]
    st.markdown(
        f"**Carries real signal:** `{'`, `'.join(signal) if signal else 'none'}`  \n"
        f"**Indistinguishable from noise:** `{'`, `'.join(noise) if noise else 'none'}`"
    )

    st.divider()
    st.subheader("Ablation — what are the macro features actually worth?")

    st.dataframe(
        extras.ablation.style.format({"RMSE": "{:,.2f}", "R2": "{:.5f}"}),
        **STRETCH,
    )

    if len(extras.ablation) == 2:
        r2_solo = float(extras.ablation.iloc[0]["R2"])
        r2_full = float(extras.ablation.iloc[1]["R2"])
        delta = r2_full - r2_solo
        rel = (delta / r2_solo * 100) if r2_solo != 0 else float("nan")

        a1, a2, a3 = st.columns(3)
        a1.metric("R² — sales only", f"{r2_solo:.5f}")
        a2.metric("R² — all selected", f"{r2_full:.5f}")
        a3.metric("Relative gain", f"{rel:+.3f}%")

        # Verdict generated from the numbers rather than hardcoded, so it stays honest if
        # the user swaps in a dataset where the macro features do matter.
        if abs(rel) < 1.0:
            st.error(
                f"**Verdict:** the additional features add {rel:+.3f}% relative R². For "
                f"practical purposes they are worthless — this is a one-variable model "
                f"wearing a {len(selected_features)}-variable costume. Deploy the "
                f"`sales`-only version: same accuracy, one input instead of "
                f"{len(selected_features)}, five fewer ways to break.",
                icon="🎯",
            )
        else:
            st.success(
                f"**Verdict:** the additional features contribute {rel:+.3f}% relative R² "
                f"— non-trivial on this data, worth keeping.",
                icon="🎯",
            )
    else:
        st.info("Ablation needs `sales` in the feature set to compare against.")


# ======================================================================================
# Tab 4 — What-if prediction
# ======================================================================================

with tab_single:
    st.subheader("Single prediction")
    st.caption(
        "Sliders are bounded by each column's observed range and default to its median."
    )

    slider_cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(selected_features):
        col = slider_cols[i % 3]
        lo = float(df[feat].min())
        hi = float(df[feat].max())
        med = float(df[feat].median())
        step = (hi - lo) / 200 if hi > lo else 0.01
        inputs[feat] = col.slider(
            feat,
            min_value=lo,
            max_value=hi,
            value=med,
            step=step,
            help=f"Observed range {lo:,.2f} – {hi:,.2f} · median {med:,.2f}",
            key=f"slider_{feat}",
        )

    single = pd.DataFrame([inputs])

    try:
        result = U.predict_with_interval(
            estimator, single, selected_features, extras.lower_q, extras.upper_q
        )
    except U.DataValidationError as exc:
        st.error(str(exc))
        st.stop()

    point = float(result["prediction"].iloc[0])
    low = float(result["lower_90"].iloc[0])
    high = float(result["upper_90"].iloc[0])

    r1, r2 = st.columns([1, 2])

    with r1:
        st.metric("Predicted target_sales", f"{point:,.1f}")
        st.metric("90% interval", f"{low:,.0f} – {high:,.0f}")
        st.caption(f"Interval width: {high - low:,.1f}")

    with r2:
        t_min = float(df[U.TARGET].min())
        t_max = float(df[U.TARGET].max())
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=point,
                number={"valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [t_min, t_max]},
                    "bar": {"color": ACCENT, "thickness": 0.55},
                    "steps": [
                        {"range": [t_min, float(df[U.TARGET].quantile(0.25))], "color": "rgba(255,255,255,0.05)"},
                        {"range": [float(df[U.TARGET].quantile(0.25)), float(df[U.TARGET].quantile(0.75))], "color": "rgba(255,255,255,0.12)"},
                        {"range": [float(df[U.TARGET].quantile(0.75)), t_max], "color": "rgba(255,255,255,0.05)"},
                    ],
                    "threshold": {
                        "line": {"color": WARN, "width": 3},
                        "value": float(df[U.TARGET].median()),
                    },
                },
                title={"text": "Position within the observed target range"},
            )
        )
        st.plotly_chart(fig_layout(gauge, 300), **STRETCH)

    pct = float((df[U.TARGET] < point).mean() * 100)
    st.caption(
        f"Shaded band = interquartile range of observed `{U.TARGET}`; orange marker = "
        f"median. This prediction sits at the **{pct:.1f}th percentile** of observed values."
    )

    if len([f for f in selected_features if f in U.MACRO_FEATURES]) > 0:
        st.info(
            "**Try it:** drag the five macro sliders across their full range. The "
            "prediction will barely move. That is not a bug — it is the model correctly "
            "reflecting that those features carry no signal. Only `sales` moves the needle.",
            icon="🧪",
        )


# ======================================================================================
# Tab 5 — Batch prediction
# ======================================================================================

with tab_batch:
    st.subheader("Batch prediction")

    b1, b2 = st.columns([2, 1])
    with b2:
        st.download_button(
            "Download template CSV",
            data=U.template_csv(),
            file_name="batch_template.csv",
            mime="text/csv",
            **STRETCH,
        )
        st.caption("Header-only file with the exact columns expected.")

    with b1:
        batch_file = st.file_uploader(
            "Upload rows to score",
            type=["csv"],
            key="batch_uploader",
            help=f"Required columns: {', '.join(selected_features)}. An optional "
            f"`{U.TARGET}` column will be used to score accuracy.",
        )

    if batch_file is None:
        st.info("Upload a CSV to score rows in bulk.", icon="⬆️")
    else:
        try:
            raw = pd.read_csv(batch_file)
            batch = U.validate_frame(raw, require_target=False)
        except U.DataValidationError as exc:
            st.error(f"Upload rejected: {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file: {exc}")
            st.stop()

        missing = [c for c in selected_features if c not in batch.columns]
        if missing:
            st.error(f"Missing required column(s): {', '.join(missing)}")
            st.stop()

        preds = U.predict_with_interval(
            estimator, batch, selected_features, extras.lower_q, extras.upper_q
        )
        out = pd.concat([batch.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        st.success(f"Scored {len(out):,} rows.")

        # If the upload happens to carry ground truth, score it rather than ignore it.
        if U.TARGET in batch.columns and batch[U.TARGET].notna().any():
            mask = batch[U.TARGET].notna().values
            scores = U.score_predictions(
                batch.loc[mask, U.TARGET], preds.loc[mask, "prediction"]
            )
            coverage = float(
                (
                    (batch.loc[mask, U.TARGET].values >= preds.loc[mask, "lower_90"].values)
                    & (batch.loc[mask, U.TARGET].values <= preds.loc[mask, "upper_90"].values)
                ).mean()
                * 100
            )
            st.markdown("**Ground truth detected — scoring against it:**")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("MAE", f"{scores['MAE']:,.2f}")
            s2.metric("RMSE", f"{scores['RMSE']:,.2f}")
            s3.metric("R²", f"{scores['R2']:.4f}")
            s4.metric("MAPE", f"{scores['MAPE_%']:.2f}%")
            s5.metric(
                "90% interval coverage",
                f"{coverage:.1f}%",
                help="Should land near 90%. Materially below means the interval is too "
                "narrow for this data.",
            )
            out["error"] = out[U.TARGET] - out["prediction"]

        st.dataframe(out.head(200), **STRETCH, height=380)
        if len(out) > 200:
            st.caption(f"Showing first 200 of {len(out):,} rows. Full results in the download.")

        st.download_button(
            "Download scored CSV",
            data=out.to_csv(index=False),
            file_name="target_sales_predictions.csv",
            mime="text/csv",
        )


# ======================================================================================
# Tab 6 — Scenario projection
# ======================================================================================

with tab_scenario:
    st.subheader("Scenario projection")

    st.warning(
        "**This is a scenario projection, not a forecast.** The data has no time "
        "dimension, so the model cannot learn a trend. These lines show what the model "
        "predicts *if* sales follow the growth path you specify. The assumption is yours; "
        "the model only translates it into `target_sales`. Uncertainty bands cover model "
        "residual error only and exclude any error in the growth assumption itself — "
        "real-world uncertainty is wider.",
        icon="⚠️",
    )

    if not has_sales:
        st.error(
            "`sales` is not in the selected feature set. Since it is the only driver, "
            "projection is meaningless without it — re-enable it in the sidebar."
        )
    else:
        s1, s2, s3 = st.columns(3)
        start_sales = s1.number_input(
            "Starting sales",
            min_value=float(df[U.PRIMARY_FEATURE].min()),
            max_value=float(df[U.PRIMARY_FEATURE].max() * 2),
            value=float(df[U.PRIMARY_FEATURE].mean()),
            step=50.0,
            help="Defaults to the dataset mean.",
        )
        periods = s2.slider("Periods ahead", 1, 24, 12)
        show_band = s3.checkbox("Show 90% uncertainty band", value=True)

        st.markdown("**Growth rate per period, by scenario**")
        g1, g2, g3 = st.columns(3)
        growth = {
            "Pessimistic": g1.slider("Pessimistic (%)", -5.0, 10.0, -2.0, 0.25) / 100,
            "Base": g2.slider("Base (%)", -5.0, 10.0, 2.0, 0.25) / 100,
            "Optimistic": g3.slider("Optimistic (%)", -5.0, 10.0, 5.0, 0.25) / 100,
        }

        macro_drift = {}
        active_macros = [f for f in selected_features if f in U.MACRO_FEATURES]
        if active_macros:
            with st.expander("Optional: per-period drift on the macro indicators"):
                st.caption(
                    "Held at their medians unless you set a drift. Expect these to change "
                    "the output almost not at all — see the ablation on the performance tab."
                )
                drift_cols = st.columns(min(3, len(active_macros)))
                for i, feat in enumerate(active_macros):
                    spread = float(df[feat].std())
                    macro_drift[feat] = drift_cols[i % len(drift_cols)].number_input(
                        f"{feat} / period",
                        min_value=-spread,
                        max_value=spread,
                        value=0.0,
                        step=spread / 50 if spread > 0 else 0.01,
                        format="%.4f",
                        key=f"drift_{feat}",
                    )

        colours = {"Pessimistic": "#E05C5C", "Base": ACCENT, "Optimistic": "#5C9BE0"}
        fig = go.Figure()
        tables = []

        for label, rate in growth.items():
            frame = U.build_scenario_frame(
                df, selected_features, start_sales, periods, rate, macro_drift
            )
            res = U.predict_with_interval(
                estimator, frame, selected_features, extras.lower_q, extras.upper_q
            )
            periods_axis = list(frame.index)

            if show_band:
                # Upper first, then lower with fill='tonexty' — Plotly fills to the
                # previously drawn trace, so order matters here.
                fig.add_trace(
                    go.Scatter(
                        x=periods_axis,
                        y=res["upper_90"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{label} upper",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=periods_axis,
                        y=res["lower_90"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=rgba(colours[label], 0.13),
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{label} lower",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=periods_axis,
                    y=res["prediction"],
                    mode="lines+markers",
                    line=dict(color=colours[label], width=2.5),
                    name=f"{label} ({rate*100:+.2f}%/period)",
                )
            )

            t = pd.DataFrame(
                {
                    "period": periods_axis,
                    "scenario": label,
                    "projected_sales": frame[U.PRIMARY_FEATURE].values,
                    "predicted_target_sales": res["prediction"].values,
                    "lower_90": res["lower_90"].values,
                    "upper_90": res["upper_90"].values,
                }
            )
            tables.append(t)

        fig.update_xaxes(title="Period ahead", dtick=1)
        fig.update_yaxes(title="Predicted target_sales")
        st.plotly_chart(
            fig_layout(fig, 480, "Projected target_sales under three growth assumptions"),
            **STRETCH,
        )

        scenario_table = pd.concat(tables, ignore_index=True)

        view = st.selectbox(
            "Show table for", ["All"] + list(growth.keys()), key="scenario_table_view"
        )
        shown = (
            scenario_table
            if view == "All"
            else scenario_table[scenario_table["scenario"] == view]
        )
        st.dataframe(
            shown.style.format(
                {
                    "projected_sales": "{:,.1f}",
                    "predicted_target_sales": "{:,.1f}",
                    "lower_90": "{:,.1f}",
                    "upper_90": "{:,.1f}",
                }
            ),
            **STRETCH,
            height=340,
        )

        st.download_button(
            "Download scenario table",
            data=scenario_table.to_csv(index=False),
            file_name="scenario_projection.csv",
            mime="text/csv",
        )

        st.caption(
            "⚠️ Note on compounding: the band width is constant per period, but the "
            "projected sales path compounds. Later periods therefore look "
            "*proportionally* more certain than they are — the real error in a 24-period "
            "projection is dominated by whether your growth assumption is right, which "
            "this band does not measure at all."
        )

        with st.expander("What would a genuine forecast require?"):
            st.markdown(
                """
Three things this dataset does not have:

1. **Time-ordered observations** — a date, period or sequence column, with rows in
   chronological order and a known frequency.
2. **A chronological train/test split** — train on the earliest 80%, test on the most
   recent 20%. A random split leaks future information into training and makes any
   time-series model look far better than it is.
3. **Time-series cross-validation** — expanding or rolling-origin windows
   (`sklearn.model_selection.TimeSeriesSplit`), never plain k-fold.

With those in place the appropriate tools would be lag features with a gradient-booster,
or classical methods such as SARIMA and exponential smoothing, benchmarked against a naive
"tomorrow equals today" baseline. Without them, no amount of modelling produces a forecast
— only the conditional projection shown above.
                """
            )
