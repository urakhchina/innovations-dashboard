#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sales Forecasting – Product×Distributor monthly revenue to Dec-2025
Outputs: forecasts_2025.csv with columns:
key_type,key_id,product_name,distributor,sales_rep,date,yhat,yhat_lower,yhat_upper,model
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta



def ensure_monthly_index(ts):
    idx = pd.date_range(ts["ds"].min(), ts["ds"].max(), freq="MS")
    ts2 = ts.set_index("ds").reindex(idx).fillna(0.0).rename_axis("ds").reset_index()
    return ts2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Detailed CSV (line-level) with posting_date,revenue,product_name,distributor,sales_rep,canonical_code")
    p.add_argument("--outdir", default="./outputs")
    p.add_argument("--group", default="product_distributor", choices=["product_distributor","product","distributor","rep","product_rep"])
    p.add_argument("--horizon_months", type=int, default=12)  # forecast to year-end or N months
    p.add_argument("--date_col", default="posting_date")
    p.add_argument("--revenue_col", default="revenue")
    p.add_argument("--product_col", default="product_name")
    p.add_argument("--distributor_col", default="distributor")
    p.add_argument("--salesrep_col", default="sales_rep")
    return p.parse_args()

def to_month_start(s):
    dt = pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").astype(str) + "-01")

def build_monthly(df, args):
    df["month"] = to_month_start(df[args.date_col])
    g_cols = []
    if args.group in ("product_distributor","product","product_rep"):
        g_cols.append(args.product_col)
    if args.group in ("product_distributor","distributor"):
        g_cols.append(args.distributor_col)
    if args.group in ("rep","product_rep"):
        g_cols.append(args.salesrep_col)

    monthly = (df
        .assign(rev=pd.to_numeric(df[args.revenue_col], errors="coerce").fillna(0.0))
        .groupby(g_cols + ["month"], dropna=False)["rev"].sum()
        .reset_index()
        .rename(columns={"rev":"revenue"}))
    return monthly, g_cols

def forecast_one(ts_df):
    """ts_df: DataFrame with columns: ds, y (monthly). Returns df with ds,yhat,yhat_lower,yhat_upper,model."""
    # Try Prophet
    try:
        from prophet import Prophet
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(ts_df.rename(columns={"ds":"ds","y":"y"}))
        last_ds = ts_df["ds"].max()
        future = pd.DataFrame({"ds": pd.date_range(last_ds + relativedelta(months=1), last_ds + relativedelta(months=12), freq="MS")})
        fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        fc["model"] = "prophet"
        return fc
    except Exception:
        pass
    # Fallback: SARIMAX (simple)
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = ts_df.set_index("ds")["y"].asfreq("MS").fillna(0.0)
        # quick heuristic order; tune later
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        steps = 12
        pred = res.get_forecast(steps=steps)
        ci = pred.conf_int(alpha=0.05)
        idx = pd.date_range(y.index.max() + relativedelta(months=1), periods=steps, freq="MS")
        out = pd.DataFrame({
            "ds": idx,
            "yhat": pred.predicted_mean.values,
            "yhat_lower": ci.iloc[:,0].values,
            "yhat_upper": ci.iloc[:,1].values,
            "model": "sarimax"
        })
        return out
    except Exception as e:
        # If both fail, return empty
        return pd.DataFrame(columns=["ds","yhat","yhat_lower","yhat_upper","model"])

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    # 🔴 NEW: keep only paid lines (Σrevenue > 0)
    df["rev"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    df_paid = df[df["rev"] > 0].copy()
    monthly, keys = build_monthly(df, args)

    # build series per key group
    out_frames = []

    MIN_MONTHS = 3  # accept short series
    # Primary grouping (e.g., product_distributor)
    for combo, grp in monthly.groupby(keys, dropna=False):
        key_vals = combo if isinstance(combo, tuple) else (combo,)
        ts = grp[["month","revenue"]].rename(columns={"month":"ds","revenue":"y"}).copy()
        ts["ds"] = pd.to_datetime(ts["ds"])
        ts = ts.groupby("ds", as_index=False)["y"].sum().sort_values("ds")  # collapse dupes
        ts = ensure_monthly_index(ts)

        if (ts["y"] > 0).sum() < MIN_MONTHS:
            # Too sparse at product×distributor → try product-level roll-up
            if args.group == "product_distributor":
                prod_col = args.product_col
                prod_val = key_vals[0]
                ts_prod = (monthly.loc[monthly[prod_col] == prod_val, ["month","revenue"]]
                                .rename(columns={"month":"ds","revenue":"y"}))
                ts_prod["ds"] = pd.to_datetime(ts_prod["ds"])
                ts_prod = ts_prod.groupby("ds", as_index=False)["y"].sum().sort_values("ds")
                ts_prod = ensure_monthly_index(ts_prod)
                if (ts_prod["y"] > 0).sum() >= MIN_MONTHS:
                    fc = forecast_one(ts_prod)
                    if not fc.empty:
                        fc = fc.rename(columns={"ds":"date"})
                        fc[args.product_col] = prod_val
                        # mark distributor as 'ALL' to signal roll-up
                        fc[args.distributor_col] = "ALL"
                        out_frames.append(fc)
                continue  # done with this pair; skip direct forecast

        fc = forecast_one(ts)
        if fc.empty:
            continue

        fc = fc.rename(columns={"ds":"date"})
        # attach identifiers for product_distributor / other modes
        if args.group in ("product_distributor","product","product_rep"):
            fc[args.product_col] = key_vals[0]
        if args.group == "product_distributor":
            fc[args.distributor_col] = key_vals[1]
        if args.group == "distributor":
            fc[args.distributor_col] = key_vals[0]
        if args.group == "rep":
            fc[args.salesrep_col] = key_vals[0]
        if args.group == "product_rep":
            fc[args.salesrep_col] = key_vals[1]

        out_frames.append(fc)


    if not out_frames:
        print("[forecast] no series produced")
        return

    forecasts = pd.concat(out_frames, ignore_index=True)
    # keep only through year-end 2025 (or args.horizon_months if you prefer)
    forecasts = forecasts[forecasts["date"] <= pd.Timestamp("2025-12-01")]
    # add a key descriptor column
    forecasts["key_type"] = args.group
    if args.group == "product_distributor":
        forecasts["key_id"] = forecasts[args.product_col] + " | " + forecasts[args.distributor_col].astype(str)
    elif args.group == "product":
        forecasts["key_id"] = forecasts[args.product_col]
    elif args.group == "distributor":
        forecasts["key_id"] = forecasts[args.distributor_col]
    elif args.group == "rep":
        forecasts["key_id"] = forecasts[args.salesrep_col].astype(str)
    elif args.group == "product_rep":
        forecasts["key_id"] = forecasts[args.product_col] + " | rep:" + forecasts[args.salesrep_col].astype(str)

    out_path = Path(args.outdir) / "forecasts_2025.csv"
    forecasts.to_csv(out_path, index=False)
    print(f"[forecast] wrote {out_path} with {len(forecasts):,} rows")

if __name__ == "__main__":
    main()
