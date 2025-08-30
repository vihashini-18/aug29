import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.title("Car Price Prediction App")

# ---------- Load & clean ----------
df = pd.read_csv("Car details v3.csv")

def num_only(x):
    if pd.isna(x): return np.nan
    m = re.search(r"[\d\.]+", str(x))
    return float(m.group()) if m else np.nan

# Convert string-with-units to numeric
for col in ["mileage", "engine", "max_power"]:
    if col in df.columns:
        df[col] = df[col].apply(num_only)

# Use ONLY numeric features you specified
use_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
X = df[use_cols].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df['selling_price'], errors='coerce')

# Drop rows with missing target; impute X in the pipeline
mask = y.notna()
X, y = X.loc[mask], y.loc[mask]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Model: log-target to prevent negatives ----------
gb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("gb", GradientBoostingRegressor(random_state=42))
])

model = TransformedTargetRegressor(
    regressor=gb,
    func=np.log1p,        # train on log1p(y)
    inverse_func=np.expm1 # transform back -> >= 0
)

model.fit(X_train, y_train)

# Metrics
pred_test = model.predict(X_test)
r2 = r2_score(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
mae = mean_absolute_error(y_test, pred_test)

st.subheader("📊 Model Performance")
st.write(f"R²: **{r2:.3f}** | RMSE: **₹{rmse:,.0f}** | MAE: **₹{mae:,.0f}**")

# ---------- UI bounds from training data (1st–99th percentiles) ----------
q = X_train.quantile([0.01, 0.5, 0.99]).T
def b(col, default=None, is_int=False):
    lo = int(q.loc[col, 0.01]) if is_int else float(q.loc[col, 0.01])
    hi = int(q.loc[col, 0.99]) if is_int else float(q.loc[col, 0.99])
    de = int(q.loc[col, 0.50]) if (is_int and default is None) else (float(q.loc[col, 0.50]) if default is None else default)
    return lo, hi, de

st.subheader("📝 Enter Car Details (values constrained to training range)")
yl, yh, yd = b("year", is_int=True)
kl, kh, kd = b("km_driven", is_int=True)
ml, mh, md = b("mileage")
el, eh, ed = b("engine")
pl, ph, pd_ = b("max_power")
sl, sh, sd = b("seats", is_int=True)

c1, c2, c3 = st.columns(3)
with c1:
    year = st.number_input("Year", min_value=yl, max_value=yh, value=yd, step=1)
    mileage = st.number_input("Mileage (km/l)", min_value=ml, max_value=mh, value=md, step=0.1)
with c2:
    km_driven = st.number_input("Kilometers Driven", min_value=kl, max_value=kh, value=kd, step=1000)
    engine = st.number_input("Engine (CC)", min_value=el, max_value=eh, value=ed, step=50.0)
with c3:
    max_power = st.number_input("Max Power (bhp)", min_value=pl, max_value=ph, value=pd_, step=1.0)
    seats = st.number_input("Seats", min_value=sl, max_value=sh, value=sd, step=1)

# ---------- Predict ----------
if st.button("Predict Price"):
    row = pd.DataFrame([{
        'year': year,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }], columns=use_cols)

    price = float(model.predict(row)[0])
    # Safety clip (shouldn't be needed due to log transform, but keep just in case)
    price = max(0.0, price)

    # Warn if any input is outside training min/max
    warn = []
    for col in use_cols:
        if row[col].iloc[0] < X_train[col].min() or row[col].iloc[0] > X_train[col].max():
            warn.append(col)
    if warn:
        st.info(f"ℹ️ Note: {', '.join(warn)} outside training range; prediction may be less reliable.")

    st.success(f"💰 Estimated Selling Price: ₹ {price:,.0f}")
