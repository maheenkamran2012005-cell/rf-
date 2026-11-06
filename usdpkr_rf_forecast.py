# -*- coding: utf-8 -*-
"""
AI-Enabled FX Hedging Decision System for USD/PKR
Random Forest: (1) forecast macro factors, (2) forecast INTERBANK USD/PKR close.

Run:   python rf_usdpkr_fyp.py
         STEP 1 -> asks for your TRAIN DATA file (a file window opens)
         STEP 2 -> asks the MONTH to forecast
         STEP 3 -> asks the YEAR to forecast
       (these 3 questions are asked every time you run the code)

Output (folder RF_FX_Output next to your data file):
  forecast_Apr2025.xlsx   one value per factor:  Factor | Last (Jul-2024) | Forecast Apr-2025
  forecast_Apr2025.png    history + forecast point for USD/PKR
  RF_validation_metrics.xlsx   accuracy tests (for your report)

Expected data layout (like few_factor_train_data.xlsx):
  * Rows with a Date and ALL columns filled        = training history
  * OPTIONAL extra rows at the bottom with ONLY Foreign reserves filled = known future
    reserves. If they are not in the file, Random Forest predicts reserves too.
Packages are installed automatically if missing.
"""

# ----------------------------------------------------------------------------
# 0. AUTO-INSTALL PACKAGES
# ----------------------------------------------------------------------------
import sys, subprocess, importlib, os, warnings, calendar

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
            "matplotlib": "matplotlib", "openpyxl": "openpyxl"}


def ensure_packages():
    for module, pip_name in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"Installing missing package: {pip_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib
try:
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_FRACTION = 0.80          # chronological split used for validation
TUNE = True                    # small time-series grid search for the USD/PKR model
RUN_VALIDATION = True          # accuracy tests (set False for a faster run)
FACTORS = ["REER", "BRENT", "SBP", "FED", "PK_CPI", "US_CPI"]   # forecast by RF (stage 1)
LOG_VARS = ["RESERVES", "REER", "BRENT"]                        # modelled as log-changes
BASE_COLS = ["RESERVES"] + FACTORS + ["USDPKR"]
RES_CURRENT = ["RESERVES_t", "dRESERVES_t"]   # current-month reserves (unknown if reserves are predicted)

# keyword (lower-case, in column name) -> short name
COLUMN_MAP = [
    ("interbank", "USDPKR"), ("usd/pkr", "USDPKR"),
    ("foreign reserves", "RESERVES"), ("reer", "REER"), ("brent", "BRENT"),
    ("sbp", "SBP"), ("federal funds", "FED"),
    ("pakistan cpi", "PK_CPI"), ("us cpi", "US_CPI"),
    ("differential", "INFL_DIFF"), ("date", "Date"),
]
LABELS = {"RESERVES": "Foreign reserves", "REER": "REER", "BRENT": "Brent crude oil price",
          "SBP": "SBP policy rate", "FED": "Federal funds rate (mid)",
          "PK_CPI": "Pakistan CPI YoY (%)", "US_CPI": "US CPI YoY (%)",
          "USDPKR": "Interbank USD/PKR close"}


# ----------------------------------------------------------------------------
# 2. ASK FOR THE FILE
# ----------------------------------------------------------------------------
def pick_file():
    print("\nSTEP 1: Select your TRAIN DATA file (Excel or CSV) in the window that opens ...")
    path = ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select your training data file (Excel or CSV)",
            filetypes=[("Excel / CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")])
        root.destroy()
    except Exception:
        path = ""
    if not path:
        try:  # Google Colab
            from google.colab import files
            up = files.upload()
            path = list(up.keys())[0]
        except Exception:
            path = input("Enter the full path of your data file (xlsx/csv): ").strip().strip('"')
    if not path or not os.path.exists(path):
        sys.exit("No valid file selected. Exiting.")
    return path


# ----------------------------------------------------------------------------
# 3. LOAD + CLEAN
# ----------------------------------------------------------------------------
def load_data(path):
    raw = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    rename = {}
    for col in raw.columns:
        low = str(col).lower()
        for key, short in COLUMN_MAP:
            if key in low and short not in rename.values():
                rename[col] = short
                break
    df = raw.rename(columns=rename)
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Could not find columns for: {missing}\nColumns found: {list(raw.columns)}")
    df = df.dropna(how="all").reset_index(drop=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    train = df[df["USDPKR"].notna() & df["Date"].notna()].copy()
    train = train.sort_values("Date").set_index("Date")[BASE_COLS].astype(float)
    train = train.interpolate(limit_direction="both")

    future = df[df["USDPKR"].isna() & df["RESERVES"].notna()].copy()
    last = train.index[-1]
    res = future["RESERVES"].astype(float).values if len(future) else np.array([], dtype=float)
    dates = pd.DatetimeIndex([last + pd.DateOffset(months=i + 1) for i in range(len(res))])
    return train, res, dates


# ----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (lag features, time-series safe)
# ----------------------------------------------------------------------------
def change(s, name):
    """log-change for level variables, simple difference for rates / inflation."""
    return np.log(s).diff() if name in LOG_VARS else s.diff()


def stage1_features(d):
    """Features to predict each factor's change at t (uses info up to t-1 + known reserves at t)."""
    X = pd.DataFrame(index=d.index)
    for c in FACTORS + ["RESERVES"]:
        X[f"{c}_L1"] = d[c].shift(1)
    for c in FACTORS:
        ch = change(d[c], c)
        for k in (1, 2, 3):
            X[f"d{c}_L{k}"] = ch.shift(k)
    dres = change(d["RESERVES"], "RESERVES")
    X["RESERVES_t"] = d["RESERVES"]
    X["dRESERVES_t"] = dres
    X["dRESERVES_L1"] = dres.shift(1)
    X["dRESERVES_L2"] = dres.shift(2)
    X["dRESERVES_L3"] = dres.shift(3)
    return X


def s1_cols_for(f, cols):
    """Reserves model must not use current-month reserves; factor models must not use the extra lags."""
    if f == "RESERVES":
        return [c for c in cols if c not in RES_CURRENT]
    return [c for c in cols if c not in ("dRESERVES_L2", "dRESERVES_L3")]


def stage2_features(d):
    """Features to predict USD/PKR log-return at t from (forecasted) factors at t + lagged returns."""
    X = pd.DataFrame(index=d.index)
    for c in ["REER", "BRENT", "RESERVES"]:
        X[f"d{c}_t"] = change(d[c], c)
    for c in ["SBP", "FED", "PK_CPI", "US_CPI"]:
        X[f"{c}_t"] = d[c]
        X[f"d{c}_t"] = d[c].diff()
    X["REER_t"] = d["REER"]
    X["INFL_DIFF_t"] = d["PK_CPI"] - d["US_CPI"]
    X["RATE_DIFF_t"] = d["SBP"] - d["FED"]
    ret = np.log(d["USDPKR"]).diff()
    for k in (1, 2, 3):
        X[f"ret_L{k}"] = ret.shift(k)
    return X


# ----------------------------------------------------------------------------
# 5. MODEL FITTING
# ----------------------------------------------------------------------------
def tune_stage2(hist):
    X = stage2_features(hist)
    y = np.log(hist["USDPKR"]).diff()
    m = X.notna().all(axis=1) & y.notna()
    base = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)
    grid = {"max_depth": [None, 4, 8], "min_samples_leaf": [1, 3, 5],
            "max_features": [0.33, 0.6, 1.0]}
    gs = GridSearchCV(base, grid, cv=TimeSeriesSplit(n_splits=4),
                      scoring="neg_mean_squared_error", n_jobs=1)
    gs.fit(X[m], y[m])
    return gs.best_params_


def fit_models(hist, s2_params):
    f_models, s1_imp = {}, {}
    X1 = stage1_features(hist)
    for f in FACTORS + ["RESERVES"]:
        cols = s1_cols_for(f, X1.columns)
        y = change(hist[f], f)
        m = X1[cols].notna().all(axis=1) & y.notna()
        mdl = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                    random_state=RANDOM_STATE, n_jobs=1)
        mdl.fit(X1.loc[m, cols], y[m])
        f_models[f] = mdl
        s1_imp[f] = pd.Series(mdl.feature_importances_, index=cols)

    X2 = stage2_features(hist)
    y2 = np.log(hist["USDPKR"]).diff()
    m = X2.notna().all(axis=1) & y2.notna()
    t_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=1, **s2_params)
    t_model.fit(X2[m], y2[m])
    s2_imp = pd.Series(t_model.feature_importances_, index=X2.columns).sort_values(ascending=False)
    return f_models, t_model, s1_imp, s2_imp


def recursive_forecast(hist, future_res, dates, f_models, t_model):
    """Month-by-month forecast: predict factors first, then USD/PKR from those predicted factors."""
    d = hist[BASE_COLS].copy()
    lo, hi = hist[BASE_COLS].min(), hist[BASE_COLS].max()   # keep forecasts inside the range seen in training
    s1_cols = stage1_features(d).columns
    s2_cols = stage2_features(d).columns
    for dt, res in zip(dates, future_res):
        d.loc[dt, :] = np.nan
        if np.isnan(res):   # reserves not given for this month -> predict them
            Xr = stage1_features(d).iloc[[-1]][s1_cols_for("RESERVES", s1_cols)]
            res = d["RESERVES"].iloc[-2] * np.exp(f_models["RESERVES"].predict(Xr)[0])
            res = float(np.clip(res, lo["RESERVES"], hi["RESERVES"]))
        d.loc[dt, "RESERVES"] = res
        X1 = stage1_features(d).iloc[[-1]][s1_cols]
        for f in FACTORS:
            delta = f_models[f].predict(X1[s1_cols_for(f, s1_cols)])[0]
            prev = d[f].iloc[-2]
            val = prev * np.exp(delta) if f in LOG_VARS else prev + delta
            val = float(np.clip(val, lo[f], hi[f]))
            d.loc[dt, f] = val
        X2 = stage2_features(d).iloc[[-1]][s2_cols]
        r = t_model.predict(X2)[0]
        d.loc[dt, "USDPKR"] = d["USDPKR"].iloc[-2] * np.exp(r)
    out = d.iloc[len(hist):].copy()
    out["INFL_DIFF"] = out["PK_CPI"] - out["US_CPI"]
    return out


# ----------------------------------------------------------------------------
# 6. METRICS
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan}


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------
def parse_month(text):
    t = str(text).strip().lower()
    if t.isdigit() and 1 <= int(t) <= 12:
        return int(t)
    for i in range(1, 13):
        if t in (calendar.month_name[i].lower(), calendar.month_abbr[i].lower()):
            return i
    raise ValueError("Month not understood")


def ask_target(last_date):
    """Return the first day of the month to forecast (from command line or by asking)."""
    print(f"\nLast actual month in your data: {last_date:%b-%Y}")
    while True:
        try:
            m = parse_month(input("STEP 2: Which MONTH do you want to forecast? (e.g. 4 or Apr): "))
            y = int(input("STEP 3: Which YEAR do you want to forecast? (e.g. 2025): ").strip())
            target = pd.Timestamp(y, m, 1)
        except Exception:
            print("  Invalid input, please try again.")
            continue
        if target <= last_date:
            print(f"  Please choose a month AFTER {last_date:%b-%Y}.")
            continue
        return target


def main():
    path = pick_file()
    print(f"\nLoading: {path}")
    train, given_res, _ = load_data(path)
    n = len(train)
    last_date = train.index[-1]
    print(f"Training rows: {n}  ({train.index[0]:%b %Y} to {last_date:%b %Y})")

    target = ask_target(last_date)
    steps = (target.year - last_date.year) * 12 + (target.month - last_date.month)
    n_given = min(len(given_res), steps)
    future_res = np.array(list(given_res[:n_given]) + [np.nan] * (steps - n_given), dtype=float)
    if n_given == steps:
        print("Foreign reserves: taken from the future rows in your file.")
    elif n_given == 0:
        print("Foreign reserves: no future values in your file, so Random Forest predicts them too.")
    else:
        print(f"Foreign reserves: given for {n_given} future month(s) in your file; "
              f"Random Forest predicts the remaining {steps - n_given}.")
    future_dates = pd.DatetimeIndex([last_date + pd.DateOffset(months=i + 1) for i in range(steps)])
    tag = f"{target:%b-%Y}"
    print(f"Forecasting {tag}  ({steps} month(s) ahead of {last_date:%b-%Y})")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "RF_FX_Output")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = os.path.join(os.getcwd(), "RF_FX_Output")
        os.makedirs(out_dir, exist_ok=True)

    # ---- chronological 80/20 split (used for tuning + validation) ----
    split = int(n * TRAIN_FRACTION)
    tr, va = train.iloc[:split], train.iloc[split:]
    s2_params = {"max_depth": None, "min_samples_leaf": 3, "max_features": 0.6}
    if TUNE:
        print("\nTuning USD/PKR Random Forest (time-series cross-validation) ...")
        s2_params = tune_stage2(tr)
    print("Best parameters:", s2_params)

    # ---- validation (accuracy for your report) ----
    if RUN_VALIDATION:
        print("Running validation ...")
        X2_all = stage2_features(train)
        y2_all = np.log(train["USDPKR"]).diff()
        wf_pred = []
        for i in range(split, n):
            idx_hist = train.index[:i]
            Xh, yh = X2_all.loc[idx_hist], y2_all.loc[idx_hist]
            m = Xh.notna().all(axis=1) & yh.notna()
            mdl = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1, **s2_params)
            mdl.fit(Xh[m], yh[m])
            r_hat = mdl.predict(X2_all.iloc[[i]])[0]
            wf_pred.append(train["USDPKR"].iloc[i - 1] * np.exp(r_hat))
        naive_one = train["USDPKR"].shift(1).iloc[split:].values
        m_wf, m_n1 = metrics(va["USDPKR"], wf_pred), metrics(va["USDPKR"], naive_one)

        f_models_v, t_model_v, _, _ = fit_models(tr, s2_params)
        va_res = va["RESERVES"].values if len(given_res) > 0 else np.full(len(va), np.nan)
        val_fc = recursive_forecast(tr, va_res, va.index, f_models_v, t_model_v)
        m_pipe = metrics(va["USDPKR"], val_fc["USDPKR"])
        m_n2 = metrics(va["USDPKR"], np.repeat(tr["USDPKR"].iloc[-1], len(va)))
        summary = pd.DataFrame({
            "Test": ["RF one-step walk-forward (actual factors)", "Naive: last month's rate (one-step)",
                     "RF full pipeline (predicted factors, multi-step)", "Naive: flat last value (multi-step)"],
            "RMSE": [m_wf["RMSE"], m_n1["RMSE"], m_pipe["RMSE"], m_n2["RMSE"]],
            "MAE": [m_wf["MAE"], m_n1["MAE"], m_pipe["MAE"], m_n2["MAE"]],
            "MAPE_%": [m_wf["MAPE_%"], m_n1["MAPE_%"], m_pipe["MAPE_%"], m_n2["MAPE_%"]],
            "R2": [m_wf["R2"], m_n1["R2"], m_pipe["R2"], m_n2["R2"]]}).round(4)
        print("\n=== VALIDATION (USD/PKR, validation window = last 20% of your data) ===")
        print(summary.to_string(index=False))
        summary.to_excel(os.path.join(out_dir, "RF_validation_metrics.xlsx"), index=False)

    # ---- FINAL: refit on ALL training rows and forecast up to the target month ----
    print("\nFitting final model on all training data ...")
    f_models, t_model, _, _ = fit_models(train, s2_params)
    fc = recursive_forecast(train, future_res, future_dates, f_models, t_model)
    row = fc.iloc[-1]
    last = train.iloc[-1]

    table = pd.DataFrame({
        "Factor": ["Foreign reserves", "REER", "Brent Crude Oil Price", "SBP Policy rate",
                   "Federal Funds Rate (midpoint)", "Pakistan CPI Inflation YoY (%)",
                   "US CPI YoY Inflation (%)", "Inflation rate differential",
                   "INTERBANK USD/PKR - Close"],
        f"Last ({last_date:%b-%Y})": [last["RESERVES"], last["REER"], last["BRENT"], last["SBP"],
                                      last["FED"], last["PK_CPI"], last["US_CPI"],
                                      last["PK_CPI"] - last["US_CPI"], last["USDPKR"]],
        f"Forecast {tag}": [row["RESERVES"], row["REER"], row["BRENT"], row["SBP"], row["FED"],
                            row["PK_CPI"], row["US_CPI"], row["INFL_DIFF"], row["USDPKR"]],
    }).round(4)

    print(f"\n=== RANDOM FOREST FORECAST FOR {tag} ===")
    print(table.to_string(index=False))

    stem = f"forecast_{target:%b%Y}"
    xl_path = os.path.join(out_dir, stem + ".xlsx")
    with pd.ExcelWriter(xl_path, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="Sheet1", index=False)
        ws = xw.sheets["Sheet1"]
        for col, w in zip("ABC", (36, 20, 22)):
            ws.column_dimensions[col].width = w

    # ---- graph: history + forecast point ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train["USDPKR"], color="royalblue", lw=2, label="Historical USD/PKR")
    ax.plot([last_date, target], [last["USDPKR"], row["USDPKR"]], color="darkorange", lw=2,
            ls="--", label="Random Forest forecast")
    ax.scatter([target], [row["USDPKR"]], color="red", zorder=5, s=50)
    ax.annotate(f"{row['USDPKR']:.2f}", (target, row["USDPKR"]), textcoords="offset points",
                xytext=(0, 12), ha="center")
    ax.set_title(f"USD/PKR forecast for {tag} (Random Forest)")
    ax.set_xlabel("Date"); ax.set_ylabel("USD/PKR"); ax.grid(alpha=0.3); ax.legend(loc="upper left")
    fig.tight_layout()
    png_path = os.path.join(out_dir, stem + ".png")
    fig.savefig(png_path, dpi=150)

    print(f"\nDone. Files saved in: {out_dir}")
    print(f" - {stem}.xlsx")
    print(f" - {stem}.png")
    if RUN_VALIDATION:
        print(" - RF_validation_metrics.xlsx")
    if os.environ.get("DISPLAY") or sys.platform in ("win32", "darwin"):
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


if __name__ == "__main__":
    main()
