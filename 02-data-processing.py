"""
02-data-processing.py — IBM Telco Churn
========================================
Loads the raw Excel file, cleans it, engineers features,
and saves the processed dataset as a .parquet file for
fast, typed access by the training script.

"""

import warnings
warnings.filterwarnings("ignore")
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RAW_PATH    = os.path.join("data", "CustomerChurn.xlsx")
OUT_PARQUET = os.path.join("data", "telco-processed.parquet")
OUT_REPORT  = os.path.join("evaluation", "processing-report.png")

# ─── Column type definitions ──────────────────────────────────────────────────

ID_COLS = ["LoyaltyID", "Customer ID"]

TARGET_COL = "Churn"

NUM_COLS_RAW = ["Tenure", "Monthly Charges", "Total Charges"]

CAT_COLS = [
    "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]

SERVICE_COLS = [
    "Phone Service", "Multiple Lines", "Online Security",
    "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies",
]

# ─── 1. Load ──────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"  Loaded   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns  : {df.columns.tolist()}\n")
    return df


# ─── 2. Clean ─────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    df = df.copy()

    # ── Duplicate rows
    n_dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = int(n_dupes)

    # ── Drop ID columns (no predictive value)
    df.drop(columns=ID_COLS, errors="ignore", inplace=True)

    # ── Total Charges: stored as object with blanks for 0-tenure customers
    n_before = df["Total Charges"].isna().sum()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    mask = df["Total Charges"].isna()
    df.loc[mask, "Total Charges"] = (
        df.loc[mask, "Monthly Charges"] * df.loc[mask, "Tenure"]
    )
    report["total_charges_imputed"] = int(mask.sum())

    # ── Categorical nulls → "Unknown"
    cat_null_counts = {}
    for col in CAT_COLS:
        n = df[col].isna().sum()
        if n > 0:
            df[col].fillna("Unknown", inplace=True)
            cat_null_counts[col] = int(n)
    report["cat_nulls_filled"] = cat_null_counts

    # ── Binary target
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(np.int8)
    report["churn_rate"] = round(float(df[TARGET_COL].mean()), 4)

    # ── Enforce correct dtypes on categoricals
    for col in CAT_COLS:
        df[col] = df[col].astype("category")

    print(f"  After cleaning : {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df, report


# ─── 3. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Returns (df_with_features, new_feature_names, engineering_stats).

    engineering_stats contains every dataset-level statistic used during
    feature construction. Apps must reuse these exact values at inference
    time, otherwise the same customer produces different features in
    training vs. serving (e.g. a 1-row q75 collapses to the customer's
    own monthly-charge value, breaking High_Charge_New).
    """
    df = df.copy()
    new_features = []

    # ── Numeric derived features
    df["Avg_Charge_Per_Month"] = (
        df["Total Charges"] / df["Tenure"].replace(0, 1)
    ).round(4)
    new_features.append("Avg_Charge_Per_Month")

    df["Services_Count"] = df[SERVICE_COLS].apply(
        lambda r: (r == "Yes").sum(), axis=1
    ).astype(np.int8)
    new_features.append("Services_Count")

    df["Charge_Per_Service"] = (
        df["Monthly Charges"] / (df["Services_Count"] + 1)
    ).round(4)
    new_features.append("Charge_Per_Service")

    # ── Binary flag features
    df["Is_New_Customer"] = (df["Tenure"] <= 3).astype(np.int8)
    new_features.append("Is_New_Customer")

    df["Is_Long_Tenured"] = (df["Tenure"] >= 48).astype(np.int8)
    new_features.append("Is_Long_Tenured")

    df["Has_Support"] = (
        (df["Tech Support"] == "Yes") | (df["Online Security"] == "Yes")
    ).astype(np.int8)
    new_features.append("Has_Support")

    df["Has_No_Online_Services"] = (
        (df["Online Security"] == "No") &
        (df["Online Backup"] == "No") &
        (df["Device Protection"] == "No") &
        (df["Tech Support"] == "No")
    ).astype(np.int8)
    new_features.append("Has_No_Online_Services")

    df["Is_Month_To_Month"] = (df["Contract"] == "Month-to-month").astype(np.int8)
    new_features.append("Is_Month_To_Month")

    df["Uses_Paperless_AutoPay"] = (
        (df["Paperless Billing"] == "Yes") &
        (df["Payment Method"] == "Bank transfer (automatic)")
    ).astype(np.int8)
    new_features.append("Uses_Paperless_AutoPay")

    # ── Interaction feature: high charge + short tenure = high risk signal
    #
    # charge_q75 is a DATASET-LEVEL statistic. It MUST be persisted and
    # reused at inference time — recomputing it on a single-row prediction
    # dataframe collapses the threshold to that one row's value, which
    # silently breaks High_Charge_New for short-tenured customers below
    # the true 75th percentile.
    charge_q75 = float(df["Monthly Charges"].quantile(0.75))
    df["High_Charge_New"] = (
        (df["Monthly Charges"] >= charge_q75) & (df["Tenure"] <= 12)
    ).astype(np.int8)
    new_features.append("High_Charge_New")

    engineering_stats = {
        "monthly_charges_q75": charge_q75,
    }

    print(f"  Engineered {len(new_features)} new features: {new_features}")
    print(f"  Engineering stats          : {engineering_stats}")
    return df, new_features, engineering_stats


# ─── 4. Validation ────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    assert df.isnull().sum().sum() == 0, "Nulls remain after cleaning!"
    assert df[TARGET_COL].nunique() == 2, "Target is not binary!"
    assert (df["Tenure"] >= 0).all(),     "Negative tenure found!"
    assert (df["Monthly Charges"] > 0).all(), "Non-positive charges found!"
    print("  Validation: all checks passed.")


# ─── 5. Save to Parquet ───────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    import os
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved parquet : {path}  ({size_kb:.1f} KB)")


# ─── 6. Processing Report Figure ─────────────────────────────────────────────

def save_processing_report(df: pd.DataFrame, new_features: list, path: str) -> None:
    fig = plt.figure(figsize=(14, 8), facecolor="#ffffff")
    fig.suptitle("IBM Telco — Data Processing Report", fontsize=14,
                 fontweight="bold", y=0.98, color="#1a1a1a")

    gs = gridspec.GridSpec(2, 3, hspace=0.48, wspace=0.38,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    BLUE   = "#378ADD"
    AMBER  = "#EF9F27"
    GREEN  = "#639922"
    RED    = "#E24B4A"
    PURPLE = "#7F77DD"

    # ── A: Target distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["No churn", "Churn"]
    colors = [BLUE, RED]
    bars = ax1.bar(labels, counts.values, color=colors, width=0.5)
    ax1.set_title("Target distribution", fontsize=10, fontweight="bold", pad=6)
    ax1.set_ylabel("Customers", fontsize=9)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 30,
                 f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=9, color="#333")
    ax1.set_ylim(0, counts.max() * 1.22)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.tick_params(labelsize=9)

    # ── B: Tenure distribution by churn
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df.loc[df[TARGET_COL]==0, "Tenure"], bins=30, alpha=0.65,
             color=BLUE, label="No churn", density=True)
    ax2.hist(df.loc[df[TARGET_COL]==1, "Tenure"], bins=30, alpha=0.65,
             color=RED, label="Churn", density=True)
    ax2.set_title("Tenure by churn status", fontsize=10, fontweight="bold", pad=6)
    ax2.set_xlabel("Tenure (months)", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.tick_params(labelsize=9)

    # ── C: Monthly charges distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df.loc[df[TARGET_COL]==0, "Monthly Charges"], bins=30, alpha=0.65,
             color=BLUE, label="No churn", density=True)
    ax3.hist(df.loc[df[TARGET_COL]==1, "Monthly Charges"], bins=30, alpha=0.65,
             color=RED, label="Churn", density=True)
    ax3.set_title("Monthly charges by churn status", fontsize=10, fontweight="bold", pad=6)
    ax3.set_xlabel("Monthly charges ($)", fontsize=9)
    ax3.set_ylabel("Density", fontsize=9)
    ax3.legend(fontsize=8, frameon=False)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.tick_params(labelsize=9)

    # ── D: Churn rate by contract type
    ax4 = fig.add_subplot(gs[1, 0])
    contract_churn = (
        df.groupby("Contract", observed=True)[TARGET_COL]
        .mean()
        .sort_values(ascending=True) * 100
    )
    colors_c = [GREEN, AMBER, RED][:len(contract_churn)]
    bars4 = ax4.barh(contract_churn.index, contract_churn.values,
                     color=colors_c, height=0.5)
    ax4.set_title("Churn rate by contract type", fontsize=10, fontweight="bold", pad=6)
    ax4.set_xlabel("Churn rate (%)", fontsize=9)
    for bar, val in zip(bars4, contract_churn.values):
        ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9, color="#333")
    ax4.set_xlim(0, contract_churn.max() * 1.2)
    ax4.spines[["top","right"]].set_visible(False)
    ax4.tick_params(labelsize=9)

    # ── E: Engineered features — churn rate
    ax5 = fig.add_subplot(gs[1, 1])
    binary_feats = [f for f in new_features if df[f].nunique() == 2]
    churn_rates = {
        f: df.loc[df[f]==1, TARGET_COL].mean() * 100
        for f in binary_feats
    }
    feat_series = pd.Series(churn_rates).sort_values()
    bar_colors = [RED if v > 40 else AMBER if v > 25 else BLUE
                  for v in feat_series.values]
    ax5.barh(feat_series.index, feat_series.values, color=bar_colors, height=0.6)
    ax5.axvline(df[TARGET_COL].mean()*100, color="#888", lw=1,
                ls="--", label=f"Overall {df[TARGET_COL].mean()*100:.1f}%")
    ax5.set_title("Churn rate — engineered flags", fontsize=10, fontweight="bold", pad=6)
    ax5.set_xlabel("Churn rate when flag = 1 (%)", fontsize=9)
    ax5.legend(fontsize=7, frameon=False)
    ax5.spines[["top","right"]].set_visible(False)
    ax5.tick_params(labelsize=8)

    # ── F: Services count vs churn rate
    ax6 = fig.add_subplot(gs[1, 2])
    svc_churn = (
        df.groupby("Services_Count")[TARGET_COL].mean() * 100
    )
    svc_count = df.groupby("Services_Count")[TARGET_COL].count()
    scatter = ax6.scatter(
        svc_churn.index, svc_churn.values,
        s=svc_count.values / 4, c=svc_churn.values,
        cmap="RdYlGn_r", vmin=0, vmax=60, alpha=0.85, edgecolors="#fff", linewidths=0.5
    )
    ax6.plot(svc_churn.index, svc_churn.values, color="#888", lw=1, ls="--", zorder=0)
    ax6.set_title("Churn rate by services count", fontsize=10, fontweight="bold", pad=6)
    ax6.set_xlabel("Number of services subscribed", fontsize=9)
    ax6.set_ylabel("Churn rate (%)", fontsize=9)
    ax6.spines[["top","right"]].set_visible(False)
    ax6.tick_params(labelsize=9)
    plt.colorbar(scatter, ax=ax6, label="Churn %", pad=0.02).ax.tick_params(labelsize=7)

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    print(f"  Report saved  : {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════════════")
    print("  Script 1 — Data Processing")
    print("══════════════════════════════════════════════════\n")

    print("[ Step 1 ] Loading raw data …")
    df_raw = load_raw(RAW_PATH)

    print("[ Step 2 ] Cleaning …")
    df_clean, clean_report = clean(df_raw)
    print(f"  Duplicates removed      : {clean_report['duplicates_removed']}")
    print(f"  Total Charges imputed   : {clean_report['total_charges_imputed']}")
    print(f"  Churn rate              : {clean_report['churn_rate']:.1%}\n")

    print("[ Step 3 ] Engineering features …")
    df_final, new_features, eng_stats = engineer_features(df_clean)
    print()

    print("[ Step 4 ] Validating …")
    validate(df_final)
    print()

    print("[ Step 5 ] Saving parquet …")
    save_parquet(df_final, OUT_PARQUET)
    print()

    # ── Persist engineering stats for downstream consumers ──
    # Both stimulus selection (script 04) and the live Streamlit apps must
    # apply the SAME thresholds when re-engineering features, otherwise
    # predictions drift from what was scored during stimulus selection.
    import joblib
    os.makedirs("models", exist_ok=True)
    stats_path = os.path.join("models", "engineering-stats.joblib")
    joblib.dump(eng_stats, stats_path)
    print(f"  Saved engineering stats : {stats_path}  ({eng_stats})")
    print()

    print("[ Step 6 ] Saving processing report …")
    save_processing_report(df_final, new_features, OUT_REPORT)

    print(f"\n  Final dataset shape : {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
    print(f"  Numeric features    : {[c for c in df_final.select_dtypes(include='number').columns.tolist() if c != TARGET_COL]}")
    print(f"  Categorical features: {df_final.select_dtypes(include='category').columns.tolist()}")
    print("\nDone — ready for 03-models-train.py\n")


if __name__ == "__main__":
    main()