"""
03-models-train.py — IBM Telco Churn
======================================
Loads the processed parquet file from 02-data-processing.py,
trains four candidate models, tunes ALL of them fairly via
RandomizedSearchCV, selects the winner by CV ROC-AUC,
and saves the best pipeline as best-model.joblib.

Usage:
    python 02-data-processing.py          # run first
    python 03-models-train.py

"""

import warnings
warnings.filterwarnings("ignore")
import os

import numpy as np
import pandas as pd
import joblib
import time

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

IN_PARQUET = os.path.join("data", "telco-processed.parquet")
OUT_MODEL  = os.path.join("models", "best-model.joblib")
OUT_REPORT = os.path.join("evaluation", "training-report.png")

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
N_ITER       = 30   # RandomizedSearchCV iterations per model

TARGET_COL   = "Churn"


# ── 1. Load Processed Data ────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"  Loaded : {df.shape[0]:,} rows x {df.shape[1]} columns from parquet")
    return df


# ── 2. Define Feature Sets ────────────────────────────────────────────────────

def get_feature_sets(df: pd.DataFrame):
    exclude = {TARGET_COL}
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(include="category").columns
                if c not in exclude]
    print(f"  Numeric     ({len(num_cols)}): {num_cols}")
    print(f"  Categorical ({len(cat_cols)}): {cat_cols}")
    return num_cols, cat_cols


# ── 3. Train / Test Split ─────────────────────────────────────────────────────

def split(df: pd.DataFrame, num_cols, cat_cols):
    X = df[num_cols + cat_cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")
    print(f"  Churn rate — train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


# ── 4. Preprocessor ───────────────────────────────────────────────────────────

def make_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# ── 5. Model Definitions + Hyperparameter Grids ───────────────────────────────
#
#  Every model gets:
#    base  — untuned pipeline for the baseline CV pass
#    param_dist — RandomizedSearchCV search space
#
#  All param keys prefixed with "classifier__" to target the
#  pipeline step, not the preprocessor.

def get_model_configs(num_cols, cat_cols, pos_weight: float) -> dict:

    def pre():
        return make_preprocessor(num_cols, cat_cols)

    return {
        "Logistic Regression": {
            "base": Pipeline([
                ("preprocessor", pre()),
                ("classifier", LogisticRegression(
                    class_weight="balanced", max_iter=2000,
                    random_state=RANDOM_STATE
                )),
            ]),
            "param_dist": {
                "classifier__C":        [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
                "classifier__penalty":  ["l1", "l2", "elasticnet"],
                "classifier__solver":   ["saga"],
                "classifier__l1_ratio": np.linspace(0, 1, 6).tolist(),
                "classifier__max_iter": [2000],
            },
        },

        "Random Forest": {
            "base": Pipeline([
                ("preprocessor", pre()),
                ("classifier", RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE, n_jobs=-1
                )),
            ]),
            "param_dist": {
                "classifier__n_estimators":     [100, 200, 300, 500],
                "classifier__max_depth":        [4, 6, 8, 12, None],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__min_samples_split":[2, 5, 10],
                "classifier__max_features":     ["sqrt", "log2", 0.5],
                "classifier__bootstrap":        [True, False],
            },
        },

        "Gradient Boosting": {
            "base": Pipeline([
                ("preprocessor", pre()),
                ("classifier", GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                )),
            ]),
            "param_dist": {
                "classifier__n_estimators":     [100, 200, 300],
                "classifier__learning_rate":    [0.01, 0.05, 0.1, 0.2],
                "classifier__max_depth":        [3, 4, 5, 6],
                "classifier__subsample":        [0.6, 0.8, 1.0],
                "classifier__max_features":     ["sqrt", "log2", None],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },

        "XGBoost": {
            "base": Pipeline([
                ("preprocessor", pre()),
                ("classifier", XGBClassifier(
                    scale_pos_weight=pos_weight,
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    verbosity=0, n_jobs=-1,
                )),
            ]),
            "param_dist": {
                "classifier__n_estimators":     [100, 200, 300, 500],
                "classifier__learning_rate":    [0.005, 0.01, 0.05, 0.1, 0.2],
                "classifier__max_depth":        [3, 4, 5, 6],
                "classifier__subsample":        [0.6, 0.7, 0.8, 1.0],
                "classifier__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "classifier__reg_alpha":        [0, 0.01, 0.1, 0.5, 1.0],
                "classifier__reg_lambda":       [0.5, 1.0, 1.5, 2.0],
                "classifier__min_child_weight": [1, 3, 5],
            },
        },
    }


# ── 6. Baseline Cross-Validation (untuned) ────────────────────────────────────

def baseline_cv(configs: dict, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"\n{'='*60}")
    print(f"  Baseline CV (untuned) — {CV_FOLDS}-fold stratified")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'ROC-AUC':>10}  +/-   {'PR-AUC':>8}")
    print(f"  {'-'*55}")

    for name, cfg in configs.items():
        roc = cross_val_score(cfg["base"], X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1)
        pr  = cross_val_score(cfg["base"], X_train, y_train,
                              cv=cv, scoring="average_precision", n_jobs=-1)
        results[name] = {
            "roc_mean": roc.mean(), "roc_std": roc.std(),
            "pr_mean":  pr.mean(),  "pr_std":  pr.std(),
            "tuned_roc_mean": None, "tuned_roc_std": None,
            "tuned_pr_mean":  None, "best_params": None,
        }
        print(f"  {name:<25} {roc.mean():.4f}  "
              f"+-{roc.std():.4f}  {pr.mean():.4f}")

    print(f"{'='*60}")
    return results


# ── 7. Tune ALL Models Fairly ─────────────────────────────────────────────────

def tune_all(configs: dict, X_train, y_train, results: dict):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tuned_models = {}

    print(f"\n{'='*60}")
    print(f"  Tuning all models — RandomizedSearchCV (n_iter={N_ITER})")
    print(f"{'='*60}")

    for name, cfg in configs.items():
        t0 = time.time()
        print(f"\n  [{name}] searching …")

        search = RandomizedSearchCV(
            estimator            = cfg["base"],
            param_distributions  = cfg["param_dist"],
            n_iter               = N_ITER,
            scoring              = "roc_auc",
            cv                   = cv,
            refit                = True,
            random_state         = RANDOM_STATE,
            n_jobs               = -1,
            verbose              = 0,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        pr_scores = cross_val_score(
            search.best_estimator_, X_train, y_train,
            cv=cv, scoring="average_precision", n_jobs=-1
        )
        tuned_std = search.cv_results_["std_test_score"][search.best_index_]
        gain = search.best_score_ - results[name]["roc_mean"]

        results[name]["tuned_roc_mean"] = search.best_score_
        results[name]["tuned_roc_std"]  = tuned_std
        results[name]["tuned_pr_mean"]  = pr_scores.mean()
        results[name]["best_params"]    = search.best_params_
        tuned_models[name]              = search.best_estimator_

        print(f"    Baseline : {results[name]['roc_mean']:.4f}")
        print(f"    Tuned    : {search.best_score_:.4f}  (+{gain:.4f})  [{elapsed:.1f}s]")
        print(f"    Best params: {search.best_params_}")

    print(f"\n{'='*60}")
    return tuned_models, results


# ── 8. Select Best Model ──────────────────────────────────────────────────────

def select_best(tuned_models: dict, results: dict):
    best_name = max(tuned_models,
                    key=lambda n: results[n]["tuned_roc_mean"])

    print("\n  Final leaderboard (tuned ROC-AUC):")
    ranked = sorted(results.items(),
                    key=lambda x: x[1]["tuned_roc_mean"], reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        marker = "  <-- winner" if name == best_name else ""
        print(f"  {rank}. {name:<25} {r['tuned_roc_mean']:.4f} "
              f"+/-{r['tuned_roc_std']:.4f}{marker}")

    return best_name, tuned_models[best_name]


# ── 9. Final Test Evaluation ──────────────────────────────────────────────────

def evaluate(model, X_test, y_test, name="Best model"):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    roc   = roc_auc_score(y_test, proba)
    pr    = average_precision_score(y_test, proba)

    print(f"\n{'-'*60}")
    print(f"  {name} — held-out test set")
    print(f"{'-'*60}")
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}\n")
    print(classification_report(y_test, preds,
                                target_names=["No churn", "Churn"], digits=3))
    return proba, preds, roc, pr


# ── 10. Risk Tier Assignment ──────────────────────────────────────────────────

def assign_tiers(proba: np.ndarray) -> np.ndarray:
    return np.where(proba < 0.30, "Low",
           np.where(proba < 0.60, "Medium", "High"))


# ── 11. Training Report Figure ────────────────────────────────────────────────

def save_training_report(results, best_name, proba_test, y_test, preds_test, path):
    fig = plt.figure(figsize=(16, 10), facecolor="#ffffff")
    fig.suptitle("IBM Telco Churn — Model Training & Tuning Report",
                 fontsize=14, fontweight="bold", y=0.98, color="#1a1a1a")

    gs = fig.add_gridspec(2, 4, hspace=0.44, wspace=0.38,
                          left=0.06, right=0.98, top=0.92, bottom=0.07)

    palette = {
        "Logistic Regression": "#EF9F27",
        "Random Forest":       "#639922",
        "Gradient Boosting":   "#378ADD",
        "XGBoost":             "#7F77DD",
    }
    names  = list(results.keys())
    colors = [palette[n] for n in names]

    # A: Baseline vs Tuned ROC-AUC
    ax1 = fig.add_subplot(gs[0, :2])
    x      = np.arange(len(names))
    width  = 0.35
    base_v  = [results[n]["roc_mean"]       for n in names]
    tuned_v = [results[n]["tuned_roc_mean"]  for n in names]
    tuned_s = [results[n]["tuned_roc_std"]   for n in names]
    hatches = ["//", "\\\\", "xx", ".."]

    b1 = ax1.bar(x - width/2, base_v, width, color=colors, alpha=0.4,
                 label="Baseline", hatch=hatches)
    b2 = ax1.bar(x + width/2, tuned_v, width, color=colors, alpha=0.9,
                 label="Tuned",
                 yerr=tuned_s, error_kw=dict(elinewidth=1, capsize=3, ecolor="#888"))
    winner_idx = names.index(best_name)
    b2[winner_idx].set_edgecolor("#1a1a1a")
    b2[winner_idx].set_linewidth(2)
    for bar, val in zip(b2, tuned_v):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0.76, 0.90)
    ax1.axhline(0.80, color="#ccc", lw=0.8, ls="--")
    ax1.set_ylabel("ROC-AUC (5-fold CV)", fontsize=9)
    ax1.set_title("Baseline vs tuned — all models (bold = winner)",
                  fontsize=10, fontweight="bold", pad=6)
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=9)

    # B: Tuning gain
    ax2 = fig.add_subplot(gs[0, 2])
    gains = [results[n]["tuned_roc_mean"] - results[n]["roc_mean"] for n in names]
    ax2.barh(names, gains, color=colors, height=0.5)
    ax2.axvline(0, color="#888", lw=0.8)
    for i, (g, c) in enumerate(zip(gains, colors)):
        ax2.text(g + 0.0002, i, f"+{g:.4f}", va="center", fontsize=8)
    ax2.set_xlabel("Delta ROC-AUC (tuned - baseline)", fontsize=9)
    ax2.set_title("Tuning gain per model", fontsize=10, fontweight="bold", pad=6)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=9)

    # C: PR-AUC tuned
    ax3 = fig.add_subplot(gs[0, 3])
    pr_v = [results[n]["tuned_pr_mean"] for n in names]
    bars3 = ax3.barh(names, pr_v, color=colors, height=0.5)
    for bar, val in zip(bars3, pr_v):
        ax3.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8)
    ax3.set_xlabel("PR-AUC (5-fold CV, tuned)", fontsize=9)
    ax3.set_title("PR-AUC — tuned models", fontsize=10, fontweight="bold", pad=6)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(labelsize=9)

    # D: Confusion matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, preds_test)
    ax4.imshow(cm, cmap="Blues")
    lbls = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, f"{lbls[i][j]}\n{cm[i,j]:,}", ha="center", va="center",
                     fontsize=10, color="white" if cm[i,j] > cm.max()/2 else "#333")
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Pred No", "Pred Yes"], fontsize=8)
    ax4.set_yticklabels(["Actual No", "Actual Yes"], fontsize=8)
    ax4.set_title(f"Confusion matrix\n({best_name})", fontsize=10, fontweight="bold", pad=6)

    # E: Score distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(proba_test[y_test == 0], bins=30, alpha=0.65,
             color="#378ADD", label="No churn", density=True)
    ax5.hist(proba_test[y_test == 1], bins=30, alpha=0.65,
             color="#E24B4A", label="Churn",    density=True)
    ax5.axvline(0.30, color="#888", lw=1, ls="--", label="Thresholds")
    ax5.axvline(0.60, color="#555", lw=1, ls="--")
    ax5.set_xlabel("Predicted probability", fontsize=9)
    ax5.set_ylabel("Density", fontsize=9)
    ax5.set_title("Score distribution by class", fontsize=10, fontweight="bold", pad=6)
    ax5.legend(fontsize=7, frameon=False)
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.tick_params(labelsize=9)

    # F: Risk tier distribution
    ax6 = fig.add_subplot(gs[1, 2])
    tiers  = assign_tiers(proba_test)
    tcnts  = pd.Series(tiers).value_counts().reindex(["Low", "Medium", "High"])
    tclrs  = ["#639922", "#EF9F27", "#E24B4A"]
    bars6  = ax6.bar(tcnts.index, tcnts.values, color=tclrs, width=0.5)
    total  = len(tiers)
    for bar, val in zip(bars6, tcnts.values):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 3,
                 f"{val}\n({val/total:.0%})", ha="center", fontsize=8)
    ax6.set_ylim(0, tcnts.max() * 1.25)
    ax6.set_ylabel("Customers", fontsize=9)
    ax6.set_title("Risk tier distribution\n(test set)", fontsize=10, fontweight="bold", pad=6)
    ax6.spines[["top", "right"]].set_visible(False)
    ax6.tick_params(labelsize=9)

    # G: Lift by decile
    ax7 = fig.add_subplot(gs[1, 3])
    decile = pd.qcut(proba_test, q=10, labels=False, duplicates="drop") + 1
    grp = pd.DataFrame({"d": decile, "c": y_test.values}).groupby("d")["c"].mean() * 100
    ax7.bar(grp.index, grp.values, color="#7F77DD", width=0.7)
    ax7.axhline(y_test.mean() * 100, color="#E24B4A", lw=1, ls="--",
                label=f"Overall {y_test.mean()*100:.1f}%")
    ax7.set_xlabel("Score decile (1=lowest, 10=highest risk)", fontsize=9)
    ax7.set_ylabel("Actual churn rate (%)", fontsize=9)
    ax7.set_title("Lift by score decile", fontsize=10, fontweight="bold", pad=6)
    ax7.set_xticks(range(1, 11))
    ax7.legend(fontsize=7, frameon=False)
    ax7.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax7.spines[["top", "right"]].set_visible(False)
    ax7.tick_params(labelsize=9)

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    print(f"  Report saved : {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════════════")
    print("  Script 2 — Model Training & Fair Tuning")
    print("══════════════════════════════════════════════════\n")

    print("[ Step 1 ] Loading processed data …")
    df = load_parquet(IN_PARQUET)
    print()

    print("[ Step 2 ] Defining feature sets …")
    num_cols, cat_cols = get_feature_sets(df)
    print()

    print("[ Step 3 ] Splitting train / test …")
    X_train, X_test, y_train, y_test = split(df, num_cols, cat_cols)
    neg, pos  = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight = round(neg / pos, 2)
    print(f"  XGBoost scale_pos_weight = {pos_weight}\n")

    print("[ Step 4 ] Building model configs …")
    configs = get_model_configs(num_cols, cat_cols, pos_weight)
    print(f"  {len(configs)} models: {list(configs.keys())}\n")

    print("[ Step 5 ] Baseline cross-validation …")
    results = baseline_cv(configs, X_train, y_train)

    print("\n[ Step 6 ] Tuning ALL models fairly …")
    tuned_models, results = tune_all(configs, X_train, y_train, results)

    print("\n[ Step 7 ] Selecting winner …")
    best_name, best_model = select_best(tuned_models, results)

    print("\n[ Step 8 ] Final evaluation on held-out test set …")
    best_model.fit(X_train, y_train)
    proba_test, preds_test, roc, pr = evaluate(best_model, X_test, y_test, best_name)

    tiers  = assign_tiers(proba_test)
    tier_df = pd.DataFrame({
        "churn_prob":   proba_test,
        "risk_tier":    tiers,
        "actual_churn": y_test.values,
    })
    print("  Risk tier summary (test set):")
    print(tier_df.groupby("risk_tier").agg(
        customers         =("churn_prob",   "count"),
        avg_prob          =("churn_prob",   "mean"),
        actual_churn_rate =("actual_churn", "mean"),
    ).round(3).to_string())

    print("\n[ Step 9 ] Saving model …")
    joblib.dump(best_model, OUT_MODEL)
    print(f"  Saved : {OUT_MODEL}")

    print("\n[ Step 10 ] Saving training report …")
    save_training_report(results, best_name, proba_test, y_test, preds_test, OUT_REPORT)


    # ── Step 11: Save SHAP artifacts for dashboard use ──────────────────────────
    # Saves three files so dashboards can render the global SHAP chart and
    # local explanations without requiring the full training dataset at runtime.
    #
    # Files saved:
    #   shap_background.npy      — preprocessed background matrix (150 rows)
    #   shap_global_values.npy   — SHAP values for those 150 rows
    #   shap_feature_names.json  — transformed feature names after OHE
    #
    # Usage in dashboard (utils.py render_global_shap_panel):
    #   X_bg    = np.load("models/shap_background.npy")
    #   sv      = np.load("models/shap_global_values.npy")
    #   names   = json.load(open("models/shap_feature_names.json"))
    print("\n[ Step 11 ] Saving SHAP background artifacts …")
    try:
        import shap as _shap
        import numpy as _np
        import json as _json

        pre       = best_model.named_steps["preprocessor"]
        clf       = best_model.named_steps["classifier"]

        # Sample 150 rows from training set as background
        X_bg_raw  = X_train.sample(min(150, len(X_train)), random_state=42)
        X_bg_trans = pre.transform(X_bg_raw)

        # Feature names after OHE
        ohe_names  = pre.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        feat_names = num_cols + ohe_names

        # Compute global SHAP values (TreeExplainer for GBM/RF/XGB)
        explainer  = _shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X_bg_trans)

        out_bg    = OUT_MODEL.replace("best-model.joblib", "shap_background.npy")
        out_sv    = OUT_MODEL.replace("best-model.joblib", "shap_global_values.npy")
        out_names = OUT_MODEL.replace("best-model.joblib", "shap_feature_names.json")

        _np.save(out_bg,  X_bg_trans)
        _np.save(out_sv,  shap_vals)
        with open(out_names, "w") as _f:
            _json.dump(feat_names, _f)

        print(f"  Background matrix  : {out_bg}  shape={X_bg_trans.shape}")
        print(f"  Global SHAP values : {out_sv}  shape={_np.array(shap_vals).shape}")
        print(f"  Feature names      : {out_names}  ({len(feat_names)} features)")

        # Also save model_columns.joblib so dashboards can align feature inputs
        out_cols = OUT_MODEL.replace("best-model.joblib", "model-columns.joblib")
        joblib.dump(list(X_train.columns), out_cols)
        print(f"  Model columns      : {out_cols}")

    except Exception as _e:
        print(f"  ⚠ SHAP artifacts skipped: {_e}")
        print("    (Install shap and re-run to generate global SHAP artifacts.)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()