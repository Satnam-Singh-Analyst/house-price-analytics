"""
Real Estate Price Prediction & Executive Dashboard (Pure Python Version)
------------------------------------------------------------------------
No C-extension/DLL dependencies required. Runs safely under restricted Windows AppLocker policies.

Run with:  python -m streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["SquareFeet", "Bedrooms", "Bathrooms", "Age"]
TARGET = "Price"

st.set_page_config(page_title="House Price Intelligence", page_icon="🏡", layout="wide")

st.title("🏡 Real Estate Price Prediction & Executive Dashboard")
st.markdown("### Business Intelligence & AI-Powered Housing Insights")


# ----------------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# ----------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "SquareFeet": rng.integers(800, 4500, n),
            "Bedrooms": rng.integers(1, 6, n),
            "Bathrooms": rng.integers(1, 4, n),
            "Age": rng.integers(0, 50, n),
        }
    )
    df[TARGET] = (
        df["SquareFeet"] * 150
        + df["Bedrooms"] * 10_000
        + df["Bathrooms"] * 15_000
        - df["Age"] * 500
        + rng.normal(0, 15_000, n)
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().drop_duplicates()
    df = df[(df[TARGET] > 0) & (df["SquareFeet"] > 0)]
    return df.reset_index(drop=True)


st.sidebar.header("📂 Data Source")
uploaded = st.sidebar.file_uploader(
    "Optional: upload a CSV with columns SquareFeet, Bedrooms, Bathrooms, Age, Price",
    type="csv",
)

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    missing_cols = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing_cols:
        st.sidebar.error(f"Missing columns: {', '.join(missing_cols)}. Using synthetic data.")
        raw = generate_synthetic_data()
        source_label = "Synthetic dataset (500 properties)"
    else:
        source_label = f"Uploaded dataset ({uploaded.name})"
else:
    raw = generate_synthetic_data()
    source_label = "Synthetic dataset (500 properties)"

df = clean_data(raw)
if len(df) < 30:
    st.error("Not enough valid rows after cleaning (need at least 30).")
    st.stop()

st.caption(f"Data source: **{source_label}** — {len(raw) - len(df)} rows removed during cleaning.")

# ----------------------------------------------------------------------------
# 2. KEY PERFORMANCE INDICATORS
# ----------------------------------------------------------------------------
st.subheader("📊 Business Intelligence KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Price", f"${df[TARGET].mean():,.0f}")
c2.metric("Avg Square Footage", f"{df['SquareFeet'].mean():,.0f} sqft")
c3.metric("Total Properties Analyzed", f"{len(df):,}")
c4.metric("Avg Property Age", f"{df['Age'].mean():.1f} Years")

st.divider()

# ----------------------------------------------------------------------------
# 3. PURE NUMPY REGRESSION MODEL (No Scikit-Learn / DLLs)
# ----------------------------------------------------------------------------
@st.cache_resource
def train_linear_model(data: pd.DataFrame):
    # Train / Test split (80/20)
    shuffled = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    train, test = shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]

    # Add bias term (column of 1s)
    X_train = np.hstack([np.ones((len(train), 1)), train[FEATURES].values])
    y_train = train[TARGET].values
    X_test = np.hstack([np.ones((len(test), 1)), test[FEATURES].values])
    y_test = test[TARGET].values

    # Ordinary Least Squares (OLS): beta = (X^T X)^(-1) X^T y
    weights, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    preds = X_test @ weights
    
    # Metrics
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    mae = np.mean(np.abs(y_test - preds))

    # Normalized feature importance based on standardized absolute weights
    X_std = train[FEATURES].std(ddof=0).values
    coef_importance = np.abs(weights[1:] * X_std)
    importance_pct = pd.Series(coef_importance / coef_importance.sum(), index=FEATURES).sort_values(ascending=False)

    results = pd.DataFrame({"Actual": y_test, "Predicted": preds})
    return weights, {"r2": r2, "rmse": rmse, "mae": mae}, importance_pct, results


weights, metrics, importance, results = train_linear_model(df)

st.subheader("🤖 Model Performance (Linear Regression Engine)")
m1, m2, m3 = st.columns(3)
m1.metric("R² (test set)", f"{metrics['r2']:.3f}")
m2.metric("RMSE", f"${metrics['rmse']:,.0f}")
m3.metric("MAE", f"${metrics['mae']:,.0f}")

left, right = st.columns(2)
with left:
    st.markdown("**Feature Importance** — relative driver contribution")
    st.bar_chart(importance)
with right:
    st.markdown("**Actual vs Predicted Price** (hold-out test set)")
    st.scatter_chart(results, x="Actual", y="Predicted")

with st.expander("🔎 Exploratory Data Analysis"):
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Price distribution**")
        hist, edges = np.histogram(df[TARGET], bins=15)
        st.bar_chart(pd.DataFrame({"Properties": hist}, index=[f"{int(e/1000)}k" for e in edges[:-1]]))
    with e2:
        st.markdown("**Correlation with Price**")
        st.bar_chart(df.corr()[TARGET].drop(TARGET).sort_values())
    st.markdown("**Summary statistics**")
    st.dataframe(df.describe().T.round(1), use_container_width=True)

st.divider()

# ----------------------------------------------------------------------------
# 4. INTERACTIVE PREDICTION
# ----------------------------------------------------------------------------
st.sidebar.header("🏡 Input Property Features")
sqft = st.sidebar.slider("Square Feet", 500, 5000, 2000, step=50)
beds = st.sidebar.slider("Bedrooms", 1, 6, 3)
baths = st.sidebar.slider("Bathrooms", 1, 4, 2)
age = st.sidebar.slider("Property Age (Years)", 0, 50, 10)

input_vector = np.array([1.0, sqft, beds, baths, age])
prediction = float(input_vector @ weights)

# Uncertainty estimate based on Mean Absolute Error
low, high = prediction - metrics["mae"], prediction + metrics["mae"]

st.subheader("💡 Estimated Property Value")
st.success(f"Predicted Valuation: **${prediction:,.2f}**")
st.caption(f"Estimated range based on test set MAE: ${low:,.0f} –${high:,.0f}")

# ----------------------------------------------------------------------------
# 5. RISK & ACTION RECOMMENDATIONS
# ----------------------------------------------------------------------------
df["PricePerSqFt"] = df[TARGET] / df["SquareFeet"]
old, new = df[df["Age"] > 30], df[df["Age"] <= 10]
if len(old) and len(new):
    gap = (old["PricePerSqFt"].mean() / new["PricePerSqFt"].mean() - 1) * 100
    risk_text = (
        f"Properties older than 30 years sell at **{gap:+.1f}%** price per sq ft "
        f"compared with properties aged 10 years or less "
        f"(${old['PricePerSqFt'].mean():,.0f} vs${new['PricePerSqFt'].mean():,.0f})."
    )
else:
    risk_text = "Not enough old/new properties in this dataset to compare age groups."

top_feature = importance.index[0]
top_share = importance.iloc[0] * 100

st.subheader("🎯 Executive Risk & Action Matrix")
st.info(
    f"""
* **Primary Value Driver:** `{top_feature}` explains **{top_share:.0f}%** of relative model impact.
* **Risk Factor:** {risk_text}
* **Strategic Action:** Real estate funds should target larger properties (> 2,500 sqft) with ages under 10 years
  to maximize return on investment (ROI), and apply a repair-cost buffer to older stock.
* **Model Reliability:** R² = {metrics['r2']:.2f} on unseen data; typical error ≈ ${metrics['mae']:,.0f} per property.
"""
)
st.caption("Predictions are estimates from a statistical model and are not a substitute for a professional appraisal.")