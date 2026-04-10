
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HW_AVAILABLE = True
except Exception:
    HW_AVAILABLE = False

st.set_page_config(page_title="AI Demand Forecasting Tool", layout="wide")

def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")

def standardize_columns(df, date_col, target_col, group_col=None):
    data = df.copy()
    keep_cols = [date_col, target_col]
    rename_map = {date_col: "date", target_col: "sales"}
    if group_col:
        keep_cols.append(group_col)
        rename_map[group_col] = "group"
    data = data[keep_cols].rename(columns=rename_map)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")
    data = data.dropna(subset=["date", "sales"]).sort_values("date")
    return data

def clean_series(series, fill_method="interpolate", clip_outliers=True):
    s = series.copy().astype(float)
    if clip_outliers and len(s) >= 8:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        s = s.clip(lower=low, upper=high)
    if fill_method == "interpolate":
        s = s.interpolate(limit_direction="both")
    elif fill_method == "forward_fill":
        s = s.ffill().bfill()
    elif fill_method == "zero":
        s = s.fillna(0)
    return s

def aggregate_data(data, frequency):
    return data.set_index("date").resample(frequency)["sales"].sum().reset_index()

def make_lag_features(series, n_lags=6):
    df = pd.DataFrame({"sales": series.values})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    df["rolling_mean_3"] = df["sales"].shift(1).rolling(3).mean()
    df["rolling_mean_6"] = df["sales"].shift(1).rolling(6).mean()
    df["time_idx"] = np.arange(len(df))
    return df.dropna().reset_index(drop=True)

@dataclass
class ForecastResult:
    model_name: str
    mae: float
    rmse: float
    forecast: pd.DataFrame
    fitted: pd.Series

def linear_regression_forecast(df, horizon):
    work = df.copy().reset_index(drop=True)
    work["time_idx"] = np.arange(len(work))
    X = work[["time_idx"]]
    y = work["sales"]
    split = max(3, int(len(work) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test) if len(X_test) else np.array([])
    mae = mean_absolute_error(y_test, preds) if len(X_test) else 0.0
    rmse = np.sqrt(mean_squared_error(y_test, preds)) if len(X_test) else 0.0
    future_idx = np.arange(len(work), len(work) + horizon)
    future_pred = model.predict(pd.DataFrame({"time_idx": future_idx}))
    forecast = pd.DataFrame({"sales_forecast": future_pred})
    fitted = pd.Series(model.predict(X), index=work.index)
    return ForecastResult("Linear Regression", mae, rmse, forecast, fitted)

def random_forest_forecast(df, horizon, n_lags=6):
    work = df.copy().reset_index(drop=True)
    feat = make_lag_features(work["sales"], n_lags=n_lags)
    feature_cols = [c for c in feat.columns if c != "sales"]
    X = feat[feature_cols]
    y = feat["sales"]
    split = max(3, int(len(feat) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test) if len(X_test) else np.array([])
    mae = mean_absolute_error(y_test, preds) if len(X_test) else 0.0
    rmse = np.sqrt(mean_squared_error(y_test, preds)) if len(X_test) else 0.0
    history = list(work["sales"].values.astype(float))
    future = []
    for _ in range(horizon):
        if len(history) < n_lags:
            value = float(np.mean(history))
            history.append(value)
            future.append(value)
            continue
        last = history[-n_lags:]
        row = {f"lag_{i+1}": last[-(i+1)] for i in range(n_lags)}
        row["rolling_mean_3"] = float(np.mean(history[-3:])) if len(history) >= 3 else float(np.mean(history))
        row["rolling_mean_6"] = float(np.mean(history[-6:])) if len(history) >= 6 else float(np.mean(history))
        row["time_idx"] = len(history)
        next_val = float(model.predict(pd.DataFrame([row]))[0])
        history.append(next_val)
        future.append(next_val)
    forecast = pd.DataFrame({"sales_forecast": future})
    fitted = pd.Series(index=work.index, dtype=float)
    temp_preds = model.predict(X)
    start = len(work) - len(feat)
    fitted.iloc[:start] = np.nan
    fitted.iloc[start:] = temp_preds
    return ForecastResult("Random Forest", mae, rmse, forecast, fitted)

def holt_winters_forecast(df, horizon, seasonal_periods):
    work = df.copy().reset_index(drop=True)
    y = work["sales"]
    split = max(3, int(len(work) * 0.8))
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if seasonal_periods and seasonal_periods >= 2 and len(y_train) >= seasonal_periods * 2:
        model = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(optimized=True)
    else:
        model = ExponentialSmoothing(y_train, trend="add", seasonal=None).fit(optimized=True)
    preds = model.forecast(len(y_test)) if len(y_test) else pd.Series(dtype=float)
    mae = mean_absolute_error(y_test, preds) if len(y_test) else 0.0
    rmse = np.sqrt(mean_squared_error(y_test, preds)) if len(y_test) else 0.0
    if seasonal_periods and seasonal_periods >= 2 and len(y) >= seasonal_periods * 2:
        full_model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(optimized=True)
    else:
        full_model = ExponentialSmoothing(y, trend="add", seasonal=None).fit(optimized=True)
    future_pred = full_model.forecast(horizon)
    forecast = pd.DataFrame({"sales_forecast": list(future_pred)})
    fitted = pd.Series(full_model.fittedvalues.values, index=work.index)
    return ForecastResult("Holt-Winters", mae, rmse, forecast, fitted)

def build_future_dates(last_date, horizon, freq):
    return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

def evaluate_models(df, horizon, seasonal_periods):
    results = []
    if len(df) >= 6:
        results.append(linear_regression_forecast(df, horizon))
    if len(df) >= 12:
        results.append(random_forest_forecast(df, horizon, n_lags=min(6, max(2, len(df)//4))))
    if HW_AVAILABLE and len(df) >= 8:
        results.append(holt_winters_forecast(df, horizon, seasonal_periods))
    if not results:
        raise ValueError("Not enough data points. Please upload a longer sales history.")
    return sorted(results, key=lambda x: x.rmse)

st.title("AI-Based Demand Forecasting Tool for Small Businesses")
st.caption("Upload historical sales data, compare forecasting models, and download the best forecast.")

with st.sidebar:
    st.header("Input settings")
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    demo = st.checkbox("Use demo dataset instead", value=False)
    frequency = st.selectbox("Forecast frequency", [("Daily", "D"), ("Weekly", "W"), ("Monthly", "M")], format_func=lambda x: x[0])
    horizon = st.slider("Forecast horizon", 3, 24, 6)
    fill_method = st.selectbox("Missing value handling", ["interpolate", "forward_fill", "zero"])
    clip_outliers = st.checkbox("Clip outliers", value=True)
    st.info("Expected input: one date column and one sales column. Optional product/category column is supported.")

def make_demo_data():
    rng = pd.date_range("2024-01-01", periods=24, freq="M")
    np.random.seed(42)
    base = 200 + np.linspace(0, 60, len(rng))
    season = 35 * np.sin(np.arange(len(rng)) * 2 * np.pi / 12)
    noise = np.random.normal(0, 12, len(rng))
    sales = np.maximum(30, base + season + noise).round(0)
    return pd.DataFrame({"OrderDate": rng, "Sales": sales})

raw = make_demo_data() if demo else load_file(uploaded)
if raw is None:
    st.warning("Upload a file or choose the demo dataset to begin.")
    st.stop()

st.subheader("1) Raw data preview")
st.dataframe(raw.head(15), use_container_width=True)

cols = list(raw.columns)
c1, c2, c3 = st.columns(3)
with c1:
    date_col = st.selectbox("Select date column", cols)
with c2:
    target_col = st.selectbox("Select sales column", cols)
with c3:
    group_col = st.selectbox("Optional product/category column", ["None"] + cols)
group_col = None if group_col == "None" else group_col

data = standardize_columns(raw, date_col, target_col, group_col)
if group_col:
    groups = sorted(data["group"].dropna().astype(str).unique().tolist())
    chosen_group = st.selectbox("Choose product/category", groups)
    data = data[data["group"].astype(str) == chosen_group]

freq_code = frequency[1]
agg = aggregate_data(data, freq_code)
agg["sales"] = clean_series(agg["sales"], fill_method=fill_method, clip_outliers=clip_outliers)

if len(agg) < 6:
    st.error("The time series is too short after cleaning. Upload more historical records.")
    st.stop()

st.subheader("2) Cleaned and aggregated time series")
st.dataframe(agg, use_container_width=True)

seasonal_periods = {"D": 7, "W": 4, "M": 12}.get(freq_code, 12)
results = evaluate_models(agg, horizon, seasonal_periods)
best = results[0]

future_dates = build_future_dates(agg["date"].max(), horizon, freq_code)
forecast_df = best.forecast.copy()
forecast_df["date"] = future_dates
forecast_df = forecast_df[["date", "sales_forecast"]]

st.subheader("3) Model comparison")
metrics = pd.DataFrame({
    "Model": [r.model_name for r in results],
    "MAE": [round(r.mae, 3) for r in results],
    "RMSE": [round(r.rmse, 3) for r in results],
})
st.dataframe(metrics, use_container_width=True)
st.success(f"Best model selected: {best.model_name}")

left, right = st.columns([2, 1])
with left:
    st.subheader("4) Forecast chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg["date"], agg["sales"], label="Historical Sales")
    ax.plot(agg["date"], best.fitted, label=f"{best.model_name} Fitted")
    ax.plot(forecast_df["date"], forecast_df["sales_forecast"], marker="o", label="Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Demand Forecast")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with right:
    st.subheader("5) Business interpretation")
    avg_future = forecast_df["sales_forecast"].mean()
    direction = "increasing" if forecast_df["sales_forecast"].iloc[-1] > forecast_df["sales_forecast"].iloc[0] else "stable / decreasing"
    st.write(f"Average forecasted demand: **{avg_future:.2f} units**")
    st.write(f"Expected demand direction: **{direction}**")
    st.write("Use this result for inventory planning, purchasing, and promotions.")

st.subheader("6) Download forecast")
csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("Download forecast as CSV", csv_bytes, "demand_forecast.csv", "text/csv")

with st.expander("Sample input format"):
    st.markdown("""
| OrderDate | Sales |
|---|---:|
| 2024-01-31 | 240 |
| 2024-02-29 | 221 |
| 2024-03-31 | 265 |
""")
