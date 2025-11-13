# We'll create a fully commented Python script implementing the churn/reorder prediction pipeline.
# The script will expect a CSV like the user's "products_2025_by_upc.csv" and produce predictions.
# We'll write it to /mnt/data/churn_reorder_pipeline.py so the user can download and run locally.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Account Reorder (Churn) Prediction – 60-Day Horizon
===================================================

Learning-oriented, end-to-end pipeline:
1) Load transactional lines
2) Build invoice-level events (paid invoices only)
3) Create feature snapshots as of reference dates
4) Generate labels: "reorder within 60 days?"
5) Train models (Logistic Regression + Random Forest)
6) Evaluate (AUC/PR, calibration)
7) Export latest predictions for dashboard integration

USAGE (example):
---------------
python churn_reorder_pipeline.py \
    --input /path/to/products_2025_by_upc.csv \
    --outdir ./outputs \
    --horizon_days 60 \
    --train_start 2024-10-01 \
    --train_end   2025-06-30 \
    --valid_start 2025-07-01 \
    --valid_end   2025-08-31 \
    --test_start  2025-09-01 \
    --test_end    2025-09-30

Notes:
- Default column names are inferred from your dashboard CSVs.
- If a column is missing, the script degrades gracefully when possible.
- The "account" entity is assumed to be canonical_code. If you have ship_to_code, we can switch to (canonical_code, ship_to_code).

Author: You (adapted by ChatGPT)
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Parsing & configuration
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Path to transactional LINE-LEVEL csv (e.g., products_2025_by_upc.csv)")
    p.add_argument("--outdir", type=str, default="./outputs", help="Where to write artifacts")
    p.add_argument("--horizon_days", type=int, default=60, help="Reorder horizon (days) for the label")
    p.add_argument("--train_start", type=str, default="2024-10-01")
    p.add_argument("--train_end",   type=str, default="2025-06-30")
    p.add_argument("--valid_start", type=str, default="2025-07-01")
    p.add_argument("--valid_end",   type=str, default="2025-08-31")
    p.add_argument("--test_start",  type=str, default="2025-09-01")
    p.add_argument("--test_end",    type=str, default="2025-09-30")
    p.add_argument("--account_key", type=str, default="canonical_code", help="Column representing the account/door")
    p.add_argument("--date_col", type=str, default="posting_date", help="Posting/shipping date column (YYYY-MM-DD)")
    p.add_argument("--revenue_col", type=str, default="revenue", help="Revenue column")
    p.add_argument("--qty_col", type=str, default=None, help="Quantity/units column (optional)")
    p.add_argument("--distributor_col", type=str, default="distributor", help="Distributor column (optional)")
    p.add_argument("--product_col", type=str, default="product_name", help="Product name/sku column (optional)")
    p.add_argument("--shipto_col", type=str, default="ship_to_code", help="Ship-to column (optional; used for invoice grouping if present)")
    p.add_argument("--salesrep_col", type=str, default="sales_rep", help="Sales rep column (optional)")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

# -----------------------------
# Utilities
# -----------------------------

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def coalesce_series(*series_list) -> pd.Series:
    # return first non-null value
    out = None
    for s in series_list:
        if s is None: 
            continue
        if out is None:
            out = s
        else:
            mask = out.isna()
            out.loc[mask] = s.loc[mask]
    return out

def safe_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# 1. Load + Build paid invoices
# -----------------------------

def load_lines(path: str, args) -> pd.DataFrame:
    print(f"[load] reading: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    return df

def build_paid_invoices(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Group lines into invoices by (distributor, account, ship_to, date) and keep Σ(revenue) > 0.
    Produces columns: account_id, invoice_date, revenue, units, plus safe passthroughs (if not group keys).
    """
    # --- Group keys ---
    has_ship = args.shipto_col in df.columns
    group_cols = [
        args.distributor_col if args.distributor_col in df.columns else None,
        args.account_key     if args.account_key     in df.columns else None,
        args.shipto_col      if has_ship else None,
        args.date_col
    ]
    group_cols = [c for c in group_cols if c is not None]

    # --- Safe numeric fields ---
    # Revenue
    if args.revenue_col in df.columns:
        df["_rev"] = pd.to_numeric(df[args.revenue_col], errors="coerce").fillna(0.0)
    else:
        df["_rev"] = 0.0  # broadcast-safe

    # Quantity / units
    if args.qty_col and args.qty_col in df.columns:
        df["_qty"] = pd.to_numeric(df[args.qty_col], errors="coerce").fillna(0.0)
    else:
        df["_qty"] = pd.Series(0.0, index=df.index)

    # --- Choose passthroughs that are NOT group keys to avoid name collisions on reset_index ---
    candidate_passthrough = []
    for col in [args.distributor_col, args.salesrep_col, args.product_col]:
        if col and (col in df.columns):
            candidate_passthrough.append(col)
    passthrough = [c for c in candidate_passthrough if c not in group_cols]

    # --- Aggregate ---
    agg_dict = {
        "revenue": ("_rev", "sum"),
        "units":   ("_qty", "sum"),
    }
    # add safe passthroughs (not in group keys)
    for col in passthrough:
        agg_dict[col] = (col, "first")

    agg = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    # --- Filter paid only ---
    inv = agg.loc[agg["revenue"] > 0].copy()

    # --- Standardize column names ---
    # rename date col to invoice_date (if it was among group_cols)
    if args.date_col in inv.columns:
        inv.rename(columns={args.date_col: "invoice_date"}, inplace=True)
    # rename account key to account_id
    if args.account_key in inv.columns:
        inv.rename(columns={args.account_key: "account_id"}, inplace=True)
    if "account_id" not in inv.columns:
        raise ValueError(f"Account key '{args.account_key}' not found in input.")

    # ensure datetime type for invoice_date
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"], errors="coerce")

    print(f"[invoices] built {len(inv):,} paid invoices from {len(df):,} lines (Σrevenue>0).")
    return inv


# -----------------------------
# 2. Feature engineering
# -----------------------------

@dataclass
class SnapshotConfig:
    horizon_days: int = 60
    window_days_long: int = 180
    window_days_short: int = 30
    window_days_med: int = 90

def make_reference_dates(min_date: pd.Timestamp, max_date: pd.Timestamp, freq="MS") -> List[pd.Timestamp]:
    # monthly start dates between min and max
    dates = pd.date_range(min_date, max_date, freq=freq)
    return list(dates)

def cut_window(df: pd.DataFrame, end_date: pd.Timestamp, days: int) -> pd.DataFrame:
    start = end_date - pd.Timedelta(days=days)
    return df.loc[(df["invoice_date"] > start) & (df["invoice_date"] <= end_date)].copy()

def account_features_asof(inv: pd.DataFrame, asof: pd.Timestamp, cfg: SnapshotConfig) -> pd.DataFrame:
    # For each account, compute snapshot features using data up to as-of (exclusive of future)
    past = inv.loc[inv["invoice_date"] <= asof].copy()
    if past.empty:
        return pd.DataFrame(columns=["account_id"])

    # last invoice date per account
    last_date = past.groupby("account_id")["invoice_date"].max().rename("last_inv_date")
    # days since last
    dsls = (asof - last_date).dt.days.rename("days_since_last")

    # rolling windows
    w30 = cut_window(past, asof, cfg.window_days_short)
    w90 = cut_window(past, asof, cfg.window_days_med)
    w180 = cut_window(past, asof, cfg.window_days_long)

    def agg_window(w: pd.DataFrame, suffix: str):
        g = w.groupby("account_id").agg(
            orders=( "invoice_date", "count"),
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            product_breadth=(args.product_col, "nunique") if args.product_col in w.columns else ("invoice_date","count"),
            distributor_breadth=(args.distributor_col, "nunique") if args.distributor_col in w.columns else ("invoice_date","count")
        )
        g.columns = [f"{c}_{suffix}" for c in g.columns]
        return g

    g30 = agg_window(w30, "30d")
    g90 = agg_window(w90, "90d")
    g180 = agg_window(w180, "180d")

    # --- Reorder interval history over long window (median days between orders) ---
    # Build per-account day deltas between consecutive invoices inside the 180d window
    intervals = (
        w180.sort_values(["account_id", "invoice_date"])
        .groupby("account_id")["invoice_date"]
        .apply(lambda s: s.diff().dt.days)    # Series of deltas (may include NaN for first row)
        .dropna()
    )

    # Median interval (in days) per account_id
    if intervals.empty:
        inter_stats = pd.DataFrame(columns=["median_interval_180d"])
    else:
        inter_stats = (
            intervals.groupby(level=0).median()   # Series indexed by account_id
                    .rename("median_interval_180d")
                    .to_frame()
        )

    # if above failed due to dtype, compute manually
    if not isinstance(inter_stats, pd.DataFrame):
        inter_stats = pd.DataFrame({
            "median_interval_180d": intervals.groupby(level=0).median()
        })

    # simple seasonality: month as cyclical
    # assign account-level month dummies based on last invoice month
    month_last = last_date.dt.month.rename("last_month")
    cyc = pd.DataFrame({"account_id": last_date.index})
    cyc = cyc.set_index("account_id")
    cyc["sin_month"] = np.sin(2*np.pi*(month_last/12.0))
    cyc["cos_month"] = np.cos(2*np.pi*(month_last/12.0))

    # combine
    feats = (pd.DataFrame(dsls)
             .join(g30, how="left")
             .join(g90, how="left")
             .join(g180, how="left")
             .join(inter_stats, how="left")
             .join(cyc, how="left")
             .fillna(0.0))
    feats.reset_index(inplace=True)
    feats.rename(columns={"index":"account_id"}, inplace=True)

    # CRITICAL FEATURE: Days into reorder cycle (0.0 = just ordered, 1.0 = at expected reorder point, >1.0 = overdue)
    # This helps model distinguish:
    #   - Accounts with 21-day cycles at day 18 (0.86 = about to reorder soon!)
    #   - Accounts with 90-day cycles at day 18 (0.20 = just restocked, won't reorder for 2+ months)
    feats["days_into_cycle"] = feats.apply(
        lambda row: row["days_since_last"] / row["median_interval_180d"]
                    if row["median_interval_180d"] > 0
                    else 0.0,
        axis=1
    )

    # target index column
    feats["asof_date"] = asof
    return feats

def make_label(inv: pd.DataFrame, account_id: str, asof: pd.Timestamp, horizon_days: int) -> int:
    # label = did the account have another paid invoice within horizon after asof?
    horizon_end = asof + pd.Timedelta(days=horizon_days)
    fut = inv[(inv["account_id"] == account_id) & (inv["invoice_date"] > asof) & (inv["invoice_date"] <= horizon_end)]
    return int(len(fut) > 0)

def build_dataset(inv: pd.DataFrame, ref_start: pd.Timestamp, ref_end: pd.Timestamp, cfg: SnapshotConfig) -> pd.DataFrame:
    # Create monthly snapshots across the period and labels for each account present so far
    refs = make_reference_dates(ref_start, ref_end, freq="MS")
    frames = []
    for asof in refs:
        feats = account_features_asof(inv, asof, cfg)
        if feats.empty: 
            continue
        feats["label"] = feats["account_id"].apply(lambda aid: make_label(inv, aid, asof, cfg.horizon_days))
        frames.append(feats)
    if not frames:
        return pd.DataFrame()
    ds = pd.concat(frames, ignore_index=True)
    return ds

# -----------------------------
# 3. Train/evaluate
# -----------------------------

@dataclass
class DataSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]

def create_splits(inv: pd.DataFrame, args, cfg: SnapshotConfig) -> DataSplits:
    # Build datasets per time window
    train = build_dataset(inv, pd.Timestamp(args.train_start), pd.Timestamp(args.train_end), cfg)
    valid = build_dataset(inv, pd.Timestamp(args.valid_start), pd.Timestamp(args.valid_end), cfg)
    test  = build_dataset(inv, pd.Timestamp(args.test_start),  pd.Timestamp(args.test_end),  cfg)

    # choose features
    feature_cols = [c for c in train.columns if c not in ("account_id","asof_date","label")]
    return DataSplits(train=train, valid=valid, test=test, feature_cols=feature_cols)

def evaluate(probs: np.ndarray, y_true: np.ndarray, split_name: str) -> Dict[str, float]:
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true))>1 else np.nan
    ap  = average_precision_score(y_true, probs) if len(np.unique(y_true))>1 else np.nan
    return {"split": split_name, "AUC": auc, "PR_AUC": ap}

def train_models(splits: DataSplits, args):
    X_tr, y_tr = splits.train[splits.feature_cols], splits.train["label"].values
    X_va, y_va = splits.valid[splits.feature_cols], splits.valid["label"].values
    X_te, y_te = splits.test[splits.feature_cols],  splits.test["label"].values

    # Scale some models (logistic) – simple standardization
    scaler = StandardScaler(with_mean=False)  # sparse-friendly
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # Logistic Regression (baseline)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None if hasattr(LogisticRegression(), "n_jobs") else None)
    logreg.fit(X_tr_s, y_tr)
    logreg_cal = CalibratedClassifierCV(logreg, method="isotonic", cv="prefit")
    logreg_cal.fit(X_va_s, y_va)

    # Random Forest (non-linear)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.random_state,
        class_weight="balanced_subsample"
    )
    rf.fit(X_tr, y_tr)  # RF handles unscaled fine

    # Evaluate
    results = []
    for name, model, scale in [
        ("logreg", logreg_cal, True),
        ("rf", rf, False),
    ]:
        va_probs = model.predict_proba(X_va_s if scale else X_va)[:,1]
        te_probs = model.predict_proba(X_te_s if scale else X_te)[:,1]
        results.append((name, evaluate(va_probs, y_va, "valid"), evaluate(te_probs, y_te, "test")))

    return {"scaler": scaler, "logreg": logreg_cal, "rf": rf, "eval": results}

# -----------------------------
# 4. Export latest predictions for dashboard
# -----------------------------

def export_latest_predictions(inv: pd.DataFrame, model_bundle, splits: DataSplits, args, cfg: SnapshotConfig, outdir: str):
    # Use the most recent available as-of in the TEST window to create current risk scores per account
    latest_asof = pd.Timestamp(args.test_end)
    latest_feats = account_features_asof(inv, latest_asof, cfg)
    if latest_feats.empty:
        print("[export] No features at latest as-of; nothing to export.")
        return None

    # Choose the better model by valid PR_AUC
    evals = model_bundle["eval"]
    best_name = None
    best_pr = -1
    for name, v, _ in evals:
        pr = v["PR_AUC"] if v["PR_AUC"] is not np.nan else -1
        if pr > best_pr:
            best_pr = pr
            best_name = name

    model = model_bundle[best_name]
    scaler = model_bundle["scaler"]
    X_latest = latest_feats[splits.feature_cols]
    if best_name == "logreg":
        X_latest_s = scaler.transform(X_latest)
        probs = model.predict_proba(X_latest_s)[:,1]
    else:
        probs = model.predict_proba(X_latest)[:,1]

    out = latest_feats[["account_id","asof_date"]].copy()
    out["risk_prob_reorder_{}d".format(cfg.horizon_days)] = probs
    out = out.sort_values("risk_prob_reorder_{}d".format(cfg.horizon_days), ascending=False)

    ensure_dir(outdir)
    out_path = os.path.join(outdir, f"churn_predictions_{cfg.horizon_days}d.csv")
    out.to_csv(out_path, index=False)
    print(f"[export] wrote predictions: {out_path}")
    return out_path

# -----------------------------
# 5. Main
# -----------------------------

def main():
    global args  # used in some feature helpers
    args = parse_args()
    ensure_dir(args.outdir)

    lines = load_lines(args.input, args)
    invoices = build_paid_invoices(lines, args)

    # Constrain to plausible date range
    invoices = invoices.sort_values(["account_id","invoice_date"])

    cfg = SnapshotConfig(horizon_days=args.horizon_days)

    # Create time-based splits → train/valid/test
    splits = create_splits(invoices, args, cfg)
    print("[dataset] train:", splits.train.shape, "valid:", splits.valid.shape, "test:", splits.test.shape)

    # Train models + evaluate
    bundle = train_models(splits, args)
    print("[eval] Validation/Test metrics (name, valid, test):")
    for name, v, t in bundle["eval"]:
        print(f"  - {name}: valid AUC={v['AUC']:.3f}, PR_AUC={v['PR_AUC']:.3f} | test AUC={t['AUC']:.3f}, PR_AUC={t['PR_AUC']:.3f}")

    # Export latest predictions for dashboard
    path = export_latest_predictions(invoices, bundle, splits, args, cfg, args.outdir)
    if path:
        # also dump a tiny JSON "schema" with column descriptions
        meta = pd.DataFrame({
            "column": ["account_id","asof_date", f"risk_prob_reorder_{cfg.horizon_days}d"],
            "description": [
                "Account/door identifier (canonical_code by default)",
                "Snapshot date used for features",
                f"Predicted probability of at least one paid invoice within {cfg.horizon_days} days after as-of"
            ]
        })
        meta_path = os.path.join(args.outdir, "churn_predictions_readme.csv")
        meta.to_csv(meta_path, index=False)
        print(f"[export] wrote schema: {meta_path}")

if __name__ == "__main__":
    main()



