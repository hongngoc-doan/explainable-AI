"""
03_select_and_build_conditions.py — IBM Telco Churn XAI Experiment
===================================================================
Selects the best customers per risk tier and generates three
between-subjects experiment condition datasets.

EXPERIMENT OVERVIEW
-------------------
Thesis focus:  Trust, Transparency, Understanding, Actionability &
               Explanation Comprehension of XAI in churn prediction.

Conditions (between-subjects):
  A — Blackbox       : customer profile + churn probability only
  B — Standard XAI   : profile + SHAP local feature contributions
  C — Actionable XAI : profile + SHAP + pre-computed retention actions

Trial layout (5 trials per condition):
  2 Low risk  +  1 Medium risk  +  2 High risk
  Rationale: High-risk cases are where actionability differences are
  most pronounced; Medium cases test ambiguity; Low cases provide
  anchor comparisons and learning.

Customer selection (multi-criteria composite scoring):
  1. Accuracy      — probability close to the tier's archetypal centre
  2. Validity      — correct prediction + ground-truth label matches tier
  3. Reasonableness— domain-knowledge feature alignment checks

  5 trials × 3 conditions = 15 unique customers (no overlap).
  Low/High tiers: 6 ranked candidates needed (2 per condition).
  Medium tier:    3 ranked candidates needed (1 per condition).

SHAP consistency design:
  N_CONSISTENT is computed from data: for each of the 10 selected
  customers in conditions B and C, find the minimum N features needed
  to explain >= SHAP_COVERAGE_THRESHOLD of total |SHAP| magnitude,
  then take the maximum across all customers. This guarantees:
    * Every customer's explanation meets the coverage threshold
    * Every B/C row has identical column structure
    * Conditions differ in XAI TYPE only, not information quantity
      -- essential for valid between-condition comparisons

  The _human_label() function collapses OHE dummy columns to their
  parent feature BEFORE coverage is computed. Without this, a single
  conceptual feature like "Contract" would consume 3 slots (one per
  category value), producing inflated N estimates.

Output files (saved as .parquet + .csv):
  Parquet -- primary: preserves dtypes, fast reads, smaller files.
  CSV     -- companion: opens in Excel / SPSS / Unipark survey builder.

  xai_selected_customers.parquet / .csv   -- all 15 winners + scores
  cond_A_blackbox.parquet        / .csv   -- 5 rows, no SHAP/actions
  cond_B_standard_xai.parquet    / .csv   -- 5 rows + SHAP columns
  cond_C_actionable_xai.parquet  / .csv   -- 5 rows + SHAP + actions
  xai_selection_report.png                -- diagnostic figure

Usage:
    python 01_data_processing.py        # prerequisite
    python 02_train_models.py           # prerequisite
    python 03_select_and_build_conditions.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import shap

# ─── ACTION STRATEGY (Option A vs Option B) ───────────────────────────────────
# Selectable via CLI flag (--strategy A|B) or the ACTION_STRATEGY env var.
# Default = "A" (validate-and-filter the hand-written templates).
#   A  -> hand-written ACTIONS library, filtered to those that empirically
#         reduce risk for THIS customer by at least 1 percentage point.
#         Fast, deterministic, easy to audit.
#   B  -> live DiCE counterfactuals, sparsity-relaxed (multi-feature), filtered
#         to those the pipeline scores ≥1pp below baseline. Slower; requires
#         dice-ml. Closer to the "DiCE counterfactual recommendations" wording
#         in the original Experimental Design document.
ACTION_STRATEGY = os.environ.get("ACTION_STRATEGY", "A").upper()
if ACTION_STRATEGY not in ("A", "B"):
    raise SystemExit(f"ACTION_STRATEGY must be 'A' or 'B', got {ACTION_STRATEGY!r}")
MIN_RISK_DROP_PP = 0.01   # minimum drop in churn prob to keep an action

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────────

IN_PARQUET = os.path.join("data", "telco-processed.parquet")
IN_MODEL   = os.path.join("models", "best-model.joblib")
OUT_DIR    = "data"

# ── Experiment constants ───────────────────────────────────────────────────────

TARGET_COL = "Churn"

RISK_LOW_MAX = 0.30
RISK_MED_MAX = 0.60

# Trials per condition: 2 Low + 1 Medium + 2 High
TRIAL_LAYOUT = {"Low": 2, "Medium": 1, "High": 2}
N_CONDITIONS = 3

# Ranked candidates needed per tier
CANDIDATES_NEEDED = {tier: count * N_CONDITIONS for tier, count in TRIAL_LAYOUT.items()}
# Low: 6, Medium: 3, High: 6

# Condition -> which rank indices to use from each tier's sorted candidate list
# Low/High: A=[0,1], B=[2,3], C=[4,5]
# Medium:   A=[0],   B=[1],   C=[2]
COND_RANK_MAP = {
    "A-blackbox":       {"Low": [0, 1], "Medium": [0], "High": [0, 1]},
    "B-standard-xai":   {"Low": [2, 3], "Medium": [1], "High": [2, 3]},
    "C-actionable-xai": {"Low": [4, 5], "Medium": [2], "High": [4, 5]},
}

# Archetypal probability target per tier
TIER_CENTRE = {"Low": 0.15, "Medium": 0.45, "High": 0.80}

# Expected ground-truth label for each tier
TIER_EXPECTED_LABEL = {"Low": 0, "Medium": 1, "High": 1}

# SHAP coverage threshold
SHAP_COVERAGE_THRESHOLD = 0.70

# Features shown to experiment participants
DISPLAY_COLS = [
    "Senior Citizen", "Partner", "Dependents",
    "Tenure", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges",
]

# ─── Hand-written retention action LIBRARY (used by Option A) ─────────────────
#
# Each entry is (title, rationale, changes_dict). The `changes_dict` lists
# the EXACT feature mutations the action implies, so at runtime we can apply
# them deterministically and ask the model to score the result -- no
# title-keyword guessing.
#
# Note: an action's "changes" can produce ZERO actual mutations for a given
# customer if they already have that feature. Validation in
# `select_actions_option_a()` filters out those no-ops.
ACTION_LIBRARY = {
    "Low": [
        ("Renew to a two-year contract",
         "Converts the current contract to the lowest-risk type -- "
         "eliminates remaining cancellation flexibility.",
         {"Contract": "Two year"}),
        ("Activate Tech Support subscription",
         "Active support is a top protective factor; "
         "having it in place reduces the renewal-risk window.",
         {"Tech Support": "Yes"}),
        ("Switch to automatic payment",
         "Customers on bank-transfer or credit-card auto-pay remove the "
         "monthly cancellation decision point -- correlates with lower churn.",
         {"Payment Method": "Credit card (automatic)", "Paperless Billing": "Yes"}),
        ("Add Online Security + Online Backup",
         "Bundling protective services increases switching cost without "
         "altering the price experience.",
         {"Online Security": "Yes", "Online Backup": "Yes"}),
    ],
    "Medium": [
        ("Offer annual contract upgrade",
         "Month-to-month -> one-year plan directly addresses the #1 churn driver.",
         {"Contract": "One year"}),
        ("Bundle Online Security + Tech Support",
         "Adding both sticky services reduces churn probability by "
         "approximately 8 percentage points.",
         {"Online Security": "Yes", "Tech Support": "Yes"}),
        ("Move to automatic payment + paperless",
         "Removes the recurring 'should I keep this?' decision tied to "
         "manual electronic-check payments.",
         {"Payment Method": "Credit card (automatic)", "Paperless Billing": "Yes"}),
        ("Add Device Protection",
         "Boosts perceived value of the bundle without changing contract.",
         {"Device Protection": "Yes"}),
        ("Two-year contract + Online Security",
         "Strong combo: longest commitment plus stickiest add-on.",
         {"Contract": "Two year", "Online Security": "Yes"}),
    ],
    "High": [
        ("Immediate two-year contract upgrade",
         "Month-to-month contract is the strongest churn driver -- "
         "offer the longest commitment within 7 days.",
         {"Contract": "Two year"}),
        ("Bundle Online Security + Tech Support + Online Backup",
         "Stack three sticky services to maximise switching cost.",
         {"Online Security": "Yes", "Tech Support": "Yes", "Online Backup": "Yes"}),
        ("One-year contract + automatic payment",
         "Smaller commitment ask if two-year is rejected; pair with auto-pay "
         "to remove the manual decision each month.",
         {"Contract": "One year",
          "Payment Method": "Credit card (automatic)",
          "Paperless Billing": "Yes"}),
        ("Annual contract + full add-on bundle",
         "If the customer balks at two-year, present one-year locked in plus "
         "the full protective bundle (Security, Support, Backup, Device).",
         {"Contract": "One year",
          "Online Security": "Yes", "Tech Support": "Yes",
          "Online Backup": "Yes",  "Device Protection": "Yes"}),
    ],
}


# Columns the model needs to score a mutated customer row. Computed from the
# trained-pipeline's column manifest if it exists; otherwise inferred at the
# call site.
def _columns_for_inference(model, df):
    """Return the exact list of columns the model expects."""
    try:
        manifest = joblib.load(os.path.join("models", "model-columns.joblib"))
    except Exception:
        try:
            manifest = joblib.load(os.path.join("models", "model_columns.joblib"))
        except Exception:
            manifest = None
    if manifest:
        return [c for c in manifest if c != "Customer ID"]
    return [c for c in df.columns
            if c not in {"Churn", "churn_prob", "pred_churn",
                         "correct", "risk_tier"}]


# Load engineering stats once (the saved Monthly-Charges q75).
try:
    _ENG_STATS = joblib.load(os.path.join("models", "engineering-stats.joblib"))
except Exception:
    _ENG_STATS = {"monthly_charges_q75": 89.85}

_SVC_COLS = ["Phone Service", "Multiple Lines", "Online Security", "Online Backup",
             "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]


def _reengineer(d):
    """
    Recompute the engineered features from a (possibly mutated) customer
    dict, using the same logic the Streamlit app's `_engineer()` uses.

    This MUST be called whenever raw features (Contract, Online Security,
    Payment Method, etc.) have been changed by a counterfactual. Without
    it, the row still carries the OLD engineered features (Has_Support,
    Services_Count, Has_No_Online_Services, etc.), so the model scores a
    self-inconsistent vector and the projected delta differs from what the
    app computes at render time. That mismatch is exactly the "synthesised
    text says ~7.9pp but Risk-if-applied chip says −10.5%" bug.
    """
    d = dict(d)
    for c in ("Tenure", "Monthly Charges", "Total Charges"):
        if c in d:
            try:
                d[c] = float(d[c])
            except Exception:
                d[c] = 0.0
    tenure_div = d.get("Tenure", 1) or 1
    d["Avg_Charge_Per_Month"] = round(d.get("Total Charges", 0) / tenure_div, 4)
    svc_count = sum(1 for c in _SVC_COLS if str(d.get(c, "No")) == "Yes")
    d["Services_Count"]       = svc_count
    d["Charge_Per_Service"]   = round(d.get("Monthly Charges", 0) / (svc_count + 1), 4)
    d["Is_New_Customer"]      = int(d.get("Tenure", 0) <= 3)
    d["Is_Long_Tenured"]      = int(d.get("Tenure", 0) >= 48)
    d["Has_Support"] = int(
        str(d.get("Tech Support", "No")) == "Yes" or
        str(d.get("Online Security", "No")) == "Yes"
    )
    d["Has_No_Online_Services"] = int(
        str(d.get("Online Security", "No"))   == "No" and
        str(d.get("Online Backup", "No"))     == "No" and
        str(d.get("Device Protection", "No")) == "No" and
        str(d.get("Tech Support", "No"))      == "No"
    )
    d["Is_Month_To_Month"]      = int(str(d.get("Contract", "")) == "Month-to-month")
    d["Uses_Paperless_AutoPay"] = int(
        str(d.get("Paperless Billing", "No")) == "Yes" and
        str(d.get("Payment Method", "")) == "Bank transfer (automatic)"
    )
    q75 = float(_ENG_STATS.get("monthly_charges_q75", 89.85))
    d["High_Charge_New"] = int(
        d.get("Monthly Charges", 0) >= q75 and d.get("Tenure", 0) <= 12
    )
    return d


def _score(model, customer_dict, expected_cols):
    """
    Score a customer dict on the live model. Re-derives the engineered
    features first so the score matches what the Streamlit app will
    compute at render time -- critical when `customer_dict` carries
    counterfactual mutations.
    """
    re = _reengineer(customer_dict)
    one = pd.DataFrame([re]).reindex(columns=expected_cols, fill_value=0)
    return float(model.predict_proba(one)[0][1])


# ─── Option A: validate-and-filter the hand-written ACTION_LIBRARY ────────────
def select_actions_option_a(row, tier, model, expected_cols, n_keep=3):
    """
    Walk the tier's library (and the others as fallback), score each action
    against THIS customer, keep only those that drop risk by at least
    MIN_RISK_DROP_PP. Return up to `n_keep` actions, sorted by best drop.
    """
    baseline = float(row.get("churn_prob", _score(model, row, expected_cols)))

    # Build a candidate pool: this tier first, then the other tiers as fallback
    # so we have something to fill remaining slots with for very-low-risk
    # customers where most actions don't move the needle.
    pool_tiers = [tier] + [t for t in ("Low", "Medium", "High") if t != tier]
    pool = []
    seen = set()
    for t in pool_tiers:
        for entry in ACTION_LIBRARY[t]:
            title = entry[0]
            if title in seen:
                continue
            seen.add(title)
            pool.append(entry)

    candidates = []
    for title, rationale, changes in pool:
        mutated = dict(row)
        mutated.update(changes)
        if mutated == row:
            continue   # no-op for this customer
        proj = _score(model, mutated, expected_cols)
        drop = baseline - proj
        if drop >= MIN_RISK_DROP_PP:
            candidates.append({"title": title, "rationale": rationale,
                               "changes": changes, "proj": proj, "drop": drop})

    candidates.sort(key=lambda c: c["drop"], reverse=True)
    return candidates[:n_keep]


# ─── Option B: live DiCE counterfactuals ──────────────────────────────────────
def select_actions_option_b(row, tier, model, expected_cols, df_population,
                            n_keep=3):
    """
    Generate counterfactuals with DiCE, filter to those that the SAME pipeline
    used in the apps scores ≥MIN_RISK_DROP_PP lower than baseline, and present
    them as actions.

    posthoc_sparsity_param=None keeps multi-feature counterfactuals (DiCE's
    default would over-simplify them). diversity_weight is bumped to encourage
    structurally different recommendations across the three slots.

    Action "title" and "rationale" are synthesised from the actual changes
    rather than hand-curated -- they're algorithmically generated so they
    track the data, not author intent.
    """
    try:
        import dice_ml
    except ImportError:
        print("    ⚠ Option B requested but `dice-ml` is not installed.")
        print("      Falling back to Option A for this customer.")
        return select_actions_option_a(row, tier, model, expected_cols, n_keep)

    baseline = float(row.get("churn_prob", _score(model, row, expected_cols)))

    # Only allow DiCE to vary features the retention specialist could actually
    # influence -- never tenure / senior-citizen / charges (those are
    # immutable customer facts, not retention levers).
    actionable_features = [
        "Contract", "Online Security", "Online Backup",
        "Device Protection", "Tech Support",
        "Paperless Billing", "Payment Method",
    ]

    # `permitted_range` restricts DiCE to retention-POSITIVE values per feature.
    # Without it, DiCE happily proposes "remove Online Security" or "set
    # Paperless Billing to No" -- those drop the predicted risk on some
    # customers but are nonsense as retention actions. We also exclude the
    # "No internet service" sentinel value from add-ons, since the experiment
    # is not asking specialists to remove customers' internet.
    permitted_range = {
        "Contract":          ["One year", "Two year"],
        "Online Security":   ["Yes"],
        "Online Backup":     ["Yes"],
        "Device Protection": ["Yes"],
        "Tech Support":      ["Yes"],
        "Paperless Billing": ["Yes"],
        "Payment Method":    ["Bank transfer (automatic)", "Credit card (automatic)"],
    }

    # Build DiCE data + model wrappers
    df_for_dice = df_population.copy()
    if "Churn" not in df_for_dice.columns and "churn_prob" in df_for_dice.columns:
        df_for_dice["Churn"] = (df_for_dice["churn_prob"] >= 0.5).astype(int)
    dice_data = dice_ml.Data(
        dataframe   = df_for_dice[expected_cols + ["Churn"]],
        continuous_features = [c for c in expected_cols
                               if pd.api.types.is_numeric_dtype(df_for_dice[c])],
        outcome_name = "Churn",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(dice_data, dice_model, method="genetic")

    query = pd.DataFrame([row]).reindex(columns=expected_cols, fill_value=0)
    try:
        cf = exp.generate_counterfactuals(
            query, total_CFs=8, desired_class=0,
            features_to_vary       = [f for f in actionable_features
                                      if f in expected_cols],
            permitted_range        = {f: vs for f, vs in permitted_range.items()
                                       if f in expected_cols},
            posthoc_sparsity_param = None,    # ← multi-feature, as requested
            proximity_weight       = 0.2,
            diversity_weight       = 2.0,
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df
    except Exception as e:
        print(f"    ⚠ DiCE generation failed ({e}); falling back to Option A.")
        return select_actions_option_a(row, tier, model, expected_cols, n_keep)

    if cf_df is None or len(cf_df) == 0:
        return select_actions_option_a(row, tier, model, expected_cols, n_keep)

    # Validate each CF against the live pipeline and synthesise an action
    candidates = []
    for _, cfrow in cf_df.iterrows():
        changes = {}
        for c in actionable_features:
            if c not in expected_cols:
                continue
            new_val = cfrow.get(c)
            if pd.notna(new_val) and str(new_val) != str(row.get(c)):
                changes[c] = new_val if not isinstance(new_val, (int, float)) \
                                     else (int(new_val) if float(new_val).is_integer()
                                                        else float(new_val))
        if not changes:
            continue
        mutated = dict(row); mutated.update(changes)
        proj = _score(model, mutated, expected_cols)
        drop = baseline - proj
        if drop < MIN_RISK_DROP_PP:
            continue
        title, rationale = _synthesise_action_text(changes, drop, customer=row)
        candidates.append({"title": title, "rationale": rationale,
                           "changes": changes, "proj": proj, "drop": drop})

    # Prefer DIVERSE counterfactuals across the three slots: greedy pick by
    # drop, but skip any that are functionally equivalent to a kept one.
    #
    # The two automatic-payment options (bank transfer / credit card) are
    # treated as the same recommendation for retention purposes -- otherwise
    # DiCE happily produces "Bundle X + Bank transfer" and "Bundle X + Credit
    # card" which look identical to a participant. The canonical_key folds
    # them into a single "AUTOPAY" token.
    def _canon_key(changes):
        canon = {}
        for k, v in changes.items():
            if k == "Payment Method" and v in ("Bank transfer (automatic)",
                                                "Credit card (automatic)"):
                canon[k] = "AUTOPAY"
            else:
                canon[k] = v
        return tuple(sorted(canon.items(), key=lambda kv: kv[0]))

    candidates.sort(key=lambda c: c["drop"], reverse=True)
    kept, kept_keys = [], set()
    for c in candidates:
        key = _canon_key(c["changes"])
        if key in kept_keys:
            continue
        kept.append(c); kept_keys.add(key)
        if len(kept) >= n_keep:
            break

    # If DiCE produced fewer than n_keep valid CFs, top up from Option A.
    if len(kept) < n_keep:
        topup = select_actions_option_a(row, tier, model, expected_cols,
                                         n_keep=n_keep - len(kept))
        for a in topup:
            key = _canon_key(a["changes"])
            if key not in kept_keys:
                kept.append(a); kept_keys.add(key)
            if len(kept) >= n_keep:
                break

    return kept[:n_keep]


def _synthesise_action_text(changes, drop, customer=None):
    """
    Turn a {feature: new_value} dict into a realistic retention-manager
    title and a customer-aware rationale. The text is anchored in
    business language ("Contract Lock-In Offer", "Service Protection
    Bundle") rather than a mechanical list of mutations, and the
    rationale references this specific customer's profile when relevant.

    Customer keys we look at: Contract, Tenure, Payment Method,
    Internet Service, Online Security, Tech Support.
    """
    customer = customer or {}
    cust_contract = str(customer.get("Contract", ""))
    cust_tenure   = customer.get("Tenure", 0)
    try:
        cust_tenure = int(cust_tenure)
    except Exception:
        cust_tenure = 0
    cust_payment  = str(customer.get("Payment Method", ""))
    cust_internet = str(customer.get("Internet Service", ""))

    # Classify what kind of action this is
    changes_contract = changes.get("Contract")
    changes_payment  = changes.get("Payment Method")
    has_security_add = any(changes.get(k) == "Yes"
                           for k in ("Online Security", "Tech Support",
                                     "Online Backup", "Device Protection"))
    n_addons = sum(1 for k in ("Online Security", "Tech Support",
                               "Online Backup", "Device Protection")
                   if changes.get(k) == "Yes")
    has_paperless = changes.get("Paperless Billing") == "Yes"

    # ── TITLE: short, retention-manager-style headline ─────────────────
    if changes_contract == "Two year" and has_security_add and changes_payment:
        title = "Premium Loyalty Bundle (Two-year contract + add-ons + auto-pay)"
    elif changes_contract == "Two year" and has_security_add:
        title = "Loyalty Lock-In (Two-year contract + protective add-ons)"
    elif changes_contract == "One year" and has_security_add and changes_payment:
        title = "Annual Retention Package (One-year contract + add-ons + auto-pay)"
    elif changes_contract == "One year" and has_security_add:
        title = "Annual Commitment Bundle (One-year contract + add-ons)"
    elif changes_contract == "Two year":
        title = "Two-Year Contract Lock-In"
    elif changes_contract == "One year":
        title = "One-Year Contract Upgrade"
    elif has_security_add and changes_payment:
        if n_addons >= 3:
            title = "Service Protection Bundle + Auto-Pay Conversion"
        else:
            title = "Targeted Add-ons + Auto-Pay Conversion"
    elif has_security_add:
        if n_addons >= 3:
            title = "Full Service Protection Bundle"
        elif n_addons == 2:
            title = "Dual Add-on Package"
        else:
            # Name the specific add-on
            for k in ("Tech Support", "Online Security",
                      "Online Backup", "Device Protection"):
                if changes.get(k) == "Yes":
                    title = f"Activate {k}"
                    break
            else:
                title = "Add Protective Service"
    elif changes_payment:
        title = "Switch to Automatic Payment"
    elif has_paperless:
        title = "Enable Paperless Billing"
    else:
        title = "Retention Adjustment"

    # ── RATIONALE: 1–2 sentences grounded in THIS customer's profile ────
    parts = []

    # Sentence 1: why these specific changes for this customer
    if changes_contract in ("One year", "Two year") and cust_contract == "Month-to-month":
        parts.append(
            f"This customer is on a month-to-month contract — the strongest "
            f"churn predictor in the model. Moving them to "
            f"{'a two-year' if changes_contract == 'Two year' else 'an annual'} "
            f"agreement removes the recurring monthly cancellation decision."
        )
    elif changes_contract == "Two year" and cust_contract == "One year":
        parts.append(
            "Extending the existing one-year contract to two years deepens "
            "the commitment and further reduces near-term churn risk."
        )
    elif has_security_add and not changes_contract:
        services_named = [k for k in ("Online Security", "Tech Support",
                                       "Online Backup", "Device Protection")
                          if changes.get(k) == "Yes"]
        names_str = " + ".join(services_named)
        parts.append(
            f"Adding {names_str} increases switching cost for this customer "
            f"without changing the contract structure. Protective add-ons "
            f"are consistently one of the strongest negative SHAP signals."
        )
    elif changes_payment and not changes_contract:
        parts.append(
            f"This customer currently pays by "
            f"{cust_payment or 'a manual method'}. Moving them to "
            f"{changes_payment} removes the monthly active-payment decision "
            f"point, which the model treats as a meaningful retention factor."
        )

    # Sentence 2: bundle rationale when contract + add-ons combine
    if changes_contract and has_security_add:
        bundle_strs = []
        for k in ("Online Security", "Tech Support",
                  "Online Backup", "Device Protection"):
            if changes.get(k) == "Yes":
                bundle_strs.append(k)
        if changes_payment:
            bundle_strs.append(f"auto-pay ({changes_payment})")
        if bundle_strs:
            parts.append(
                f"Bundling the contract change with {', '.join(bundle_strs)} "
                f"compounds the effect — each component independently lowers "
                f"churn probability in the model."
            )

    # Sentence 3: tenure-specific framing for short-tenured customers
    if 0 < cust_tenure <= 12 and (changes_contract or has_security_add):
        parts.append(
            f"With only {cust_tenure} month{'s' if cust_tenure != 1 else ''} "
            f"of tenure, this customer is still in the high-risk early "
            f"window where intervention has the largest expected impact."
        )

    # NOTE: We deliberately do NOT append a "projects a ~X pp reduction"
    # sentence here. The projected delta is already shown to the
    # participant in the "Risk if applied" chip immediately below the
    # rationale (e.g. "Risk if applied  4.5%  ↓ -10.4%"). Duplicating it
    # in prose made cards visually crowded and risked numeric mismatch if
    # rounding ever drifted between the two paths.

    # If we somehow produced no narrative sentence, add a generic line so
    # the rationale doesn't render empty.
    if not parts:
        parts.append(
            "Targeted adjustment to address this customer's strongest "
            "modifiable risk drivers identified by the model.")

    return title, " ".join(parts)


# ─── Dispatch wrapper used by build_conditions() ──────────────────────────────
def build_actions_for_customer(*, row, tier, model, num_cols, cat_cols,
                                df_population, strategy):
    """
    Single entry point that build_conditions() calls. Returns a list of
    up-to-3 action dicts of the form
        {"title": str, "rationale": str, "changes": {col: val}}
    """
    expected_cols = (num_cols or []) + (cat_cols or [])
    if not expected_cols:
        expected_cols = _columns_for_inference(model, df_population)

    if strategy == "B":
        return select_actions_option_b(row=row, tier=tier, model=model,
                                       expected_cols=expected_cols,
                                       df_population=df_population)
    return select_actions_option_a(row=row, tier=tier, model=model,
                                   expected_cols=expected_cols)


# Backwards-compat alias for code that still references ACTIONS[tier] for
# titles only (e.g. reports). Each tuple is (title, rationale).
ACTIONS = {
    tier: [(t, r) for (t, r, _ch) in entries]
    for tier, entries in ACTION_LIBRARY.items()
}


# ==============================================================================
# STEP 1 -- Load data and generate predictions
# ==============================================================================

def load_and_predict(parquet_path, model_path):
    df    = pd.read_parquet(parquet_path)
    model = joblib.load(model_path)

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET_COL]
    cat_cols = [c for c in df.select_dtypes(include="category").columns if c != TARGET_COL]

    X     = df[num_cols + cat_cols]
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    df = df.copy()
    df["churn_prob"] = proba
    df["pred_churn"] = preds
    df["correct"]    = (preds == df[TARGET_COL].values)
    df["risk_tier"]  = np.where(
        proba < RISK_LOW_MAX, "Low",
        np.where(proba < RISK_MED_MAX, "Medium", "High")
    )

    clf_name = type(model.steps[-1][1]).__name__
    print(f"  Model         : {clf_name}")
    print(f"  Customers     : {len(df):,}")
    print(f"  Churn rate    : {df[TARGET_COL].mean():.1%}\n")

    for tier in ["Low", "Medium", "High"]:
        sub = df[df["risk_tier"] == tier]
        print(f"  {tier:<8}  n={len(sub):>5,}  "
              f"actual churn={sub[TARGET_COL].mean():.1%}  "
              f"correct={sub['correct'].mean():.1%}  "
              f"prob {sub['churn_prob'].min():.3f}--{sub['churn_prob'].max():.3f}")

    return df, model, num_cols, cat_cols


# ==============================================================================
# STEP 2 -- Compute SHAP values
# ==============================================================================

def compute_shap(model, df, num_cols, cat_cols):
    """
    Compute SHAP values using TreeExplainer.
    Correct for GradientBoostingClassifier, RandomForest, and XGBoost.
    No background sample or training data required.
    """
    preprocessor  = model.named_steps["preprocessor"]
    classifier    = model.named_steps["classifier"]
    X             = df[num_cols + cat_cols]
    X_transformed = preprocessor.transform(X)
    ohe_features  = (preprocessor.named_transformers_["cat"]
                     .get_feature_names_out(cat_cols).tolist())
    feature_names = num_cols + ohe_features

    explainer  = shap.TreeExplainer(classifier)
    shap_vals  = explainer.shap_values(X_transformed)
    shap_df    = pd.DataFrame(shap_vals, index=df.index, columns=feature_names)

    print(f"  SHAP shape    : {shap_df.shape}")
    print(f"  Features      : {len(feature_names)} "
          f"({len(num_cols)} numeric + {len(ohe_features)} OHE dummies)")

    return shap_df, feature_names


# -- SHAP helper: OHE collapse + human labels ----------------------------------

def _human_label(feat):
    """
    Map a (possibly OHE) feature name to its human-readable PARENT label.

    All OHE dummies from the same original feature collapse to the same
    parent label so their SHAP contributions are summed together.
    This ensures N_CONSISTENT reflects semantic features, not
    data-encoding artefacts from one-hot encoding.
    """
    engineered = {
        "Tenure":                 "Tenure (months)",
        "Monthly Charges":        "Monthly charges ($)",
        "Total Charges":          "Total charges ($)",
        "Avg_Charge_Per_Month":   "Avg charge per month ($)",
        "Services_Count":         "Number of services",
        "Charge_Per_Service":     "Cost per service ($)",
        "Is_New_Customer":        "New customer flag",
        "Is_Long_Tenured":        "Long-tenured flag",
        "Has_Support":            "Has security/support service",
        "Has_No_Online_Services": "No online services",
        "Is_Month_To_Month":      "Month-to-month contract",
        "Uses_Paperless_AutoPay": "Paperless billing + auto-pay",
        "High_Charge_New":        "High charge + new customer",
    }
    for prefix, label in engineered.items():
        if feat == prefix or feat.startswith(prefix + "_"):
            return label

    ohe_parents = {
        "Senior Citizen":    "Senior citizen",
        "Partner":           "Has partner",
        "Dependents":        "Has dependents",
        "Phone Service":     "Phone service",
        "Multiple Lines":    "Multiple lines",
        "Internet Service":  "Internet service",
        "Online Security":   "Online security",
        "Online Backup":     "Online backup",
        "Device Protection": "Device protection",
        "Tech Support":      "Tech support",
        "Streaming TV":      "Streaming TV",
        "Streaming Movies":  "Streaming movies",
        "Contract":          "Contract type",
        "Paperless Billing": "Paperless billing",
        "Payment Method":    "Payment method",
    }
    for raw_parent, label in ohe_parents.items():
        if feat.startswith(raw_parent + "_"):
            return label

    return feat


def aggregate_shap_to_human(shap_row, n_top):
    """
    Collapse OHE dummy SHAP values to parent features (signed sum),
    then return the top n_top by absolute magnitude.
    n_top is always passed explicitly from determine_consistent_n().
    """
    agg = {}
    for feat, val in shap_row.items():
        label = _human_label(feat)
        agg[label] = agg.get(label, 0.0) + val
    ranked = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:n_top]


# ==============================================================================
# STEP 3 -- Determine N_CONSISTENT
# ==============================================================================

def determine_consistent_n(shap_df, selected_indices,
                            threshold=SHAP_COVERAGE_THRESHOLD):
    """
    N_CONSISTENT = max(minimum N to reach threshold) across all selected customers.

    Why max (not mean/median)?
      Guarantees every customer's explanation meets the coverage threshold.
      Under-explained stimuli would confound explanation comprehension
      scores -- a participant may fail the comprehension check because
      the explanation was incomplete, not because the XAI format was poor.

    Why OHE-collapsed features?
      Coverage computed on raw OHE dummies inflates N because one
      conceptual feature (Contract) consumes 3 slots. Collapsing first
      ensures the estimate reflects semantic features only.
    """
    per_customer_n = []

    print(f"\n  Coverage threshold : {threshold:.0%}")
    print(f"  {'Customer':<10} {'N needed':>9}  "
          f"{'Coverage @ N':>13}  {'Coverage @ N-1':>15}")
    print(f"  {'--'*26}")

    for idx in selected_indices:
        shap_row = shap_df.loc[idx]
        agg = {}
        for feat, val in shap_row.items():
            label = _human_label(feat)
            agg[label] = agg.get(label, 0.0) + val

        ranked    = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
        total_abs = sum(abs(v) for _, v in ranked)
        cumsum    = 0.0
        n_needed  = len(ranked)

        for n, (_, val) in enumerate(ranked, 1):
            prev_pct = cumsum / total_abs
            cumsum  += abs(val)
            curr_pct = cumsum / total_abs
            if curr_pct >= threshold:
                n_needed = n
                print(f"  C{idx:<9}  {n_needed:>9}  "
                      f"{curr_pct:>12.1%}  {prev_pct:>14.1%}")
                break

        per_customer_n.append(n_needed)

    n_consistent = max(per_customer_n)
    print(f"\n  Per-customer N values : {per_customer_n}")
    print(f"  N_CONSISTENT (max)    : {n_consistent}  "
          f"<-- applied uniformly to Conditions B and C")

    return n_consistent


# ==============================================================================
# STEP 4 -- Scoring functions
# ==============================================================================

def score_low(row):
    """
    Ideal LOW-risk profile:
      prob ~0.15 | Actual = No churn | Tenure >= 36 mo
      Annual/two-year contract | Has support service

    Low-risk cases establish the 'safe' anchor for participants.
    Long tenure and a committed contract provide clear narrative
    evidence that the model's low prediction is well-grounded.
    """
    s         = 0.0
    tolerance = 0.15

    s += max(0.0, 1.0 - abs(row["churn_prob"] - TIER_CENTRE["Low"]) / tolerance)
    if row["churn_prob"] >= 0.27:
        s -= 0.5  # penalise customers near the Low/Medium boundary
    if row["correct"]:
        s += 2.0
    if row[TARGET_COL] == 0:
        s += 1.0
    if row["Tenure"] >= 36:
        s += 1.5
    if row["Tenure"] >= 60:
        s += 0.5
    if row["Contract"] in ("One year", "Two year"):
        s += 1.5
    if row["Monthly Charges"] <= 65:
        s += 0.5
    if row["Has_Support"] == 1:
        s += 0.5
    if row["Is_New_Customer"] == 0:
        s += 0.5
    return s


def score_medium(row):
    """
    Ideal MEDIUM-risk profile:
      prob ~0.45 | Actual = Churn (preferred) | Tenure 6-36 mo
      Month-to-month contract | Mid-range charges

    Medium-risk true positives are the most instructive for XAI.
    The model's genuine uncertainty is validated by outcome, making
    the SHAP explanation meaningful for comprehension measurement.
    """
    s         = 0.0
    tolerance = 0.15

    s += max(0.0, 1.0 - abs(row["churn_prob"] - TIER_CENTRE["Medium"]) / tolerance)
    if 0.35 <= row["churn_prob"] <= 0.55:
        s += 0.5
    if row["churn_prob"] < 0.32 or row["churn_prob"] > 0.58:
        s -= 0.4
    if row["correct"]:
        s += 2.0
    if row[TARGET_COL] == 1:
        s += 1.5  # true positive preferred
    else:
        s -= 0.3
    if 6 <= row["Tenure"] <= 36:
        s += 1.5
    elif row["Tenure"] < 6:
        s -= 0.5
    elif row["Tenure"] > 48:
        s -= 0.5
    if row["Contract"] == "Month-to-month":
        s += 1.0
    if 55 <= row["Monthly Charges"] <= 85:
        s += 0.5
    if row["Has_Support"] == 0:
        s += 0.3
    return s


def score_high(row):
    """
    Ideal HIGH-risk profile:
      prob ~0.80 | Actual = Churn | Tenure <= 12 mo
      Month-to-month | No online services | Fiber optic

    Target prob = 0.80 deliberately avoids the 0.90+ ceiling.
    At extreme probabilities SHAP collapses to a single dominant
    feature, producing trivially obvious explanations that cannot
    meaningfully test explanation comprehension.
    """
    s         = 0.0
    tolerance = 0.20

    s += max(0.0, 1.0 - abs(row["churn_prob"] - TIER_CENTRE["High"]) / tolerance)
    if row["churn_prob"] > 0.90:
        s -= 0.5
    else:
        s += 0.5
    if row["correct"]:
        s += 2.0
    if row[TARGET_COL] == 1:
        s += 1.5
    if row["Tenure"] <= 12:
        s += 1.5
    if row["Tenure"] <= 6:
        s += 0.5
    if row["Contract"] == "Month-to-month":
        s += 1.0
    if row["Has_No_Online_Services"] == 1:
        s += 0.5
    if row["Internet Service"] == "Fiber optic":
        s += 0.5
    if row["Monthly Charges"] >= 65:
        s += 0.5
    return s


SCORERS = {"Low": score_low, "Medium": score_medium, "High": score_high}


# ==============================================================================
# STEP 5 -- Score, rank and select candidates
# ==============================================================================

def select_candidates(df, top_n=10):
    """
    Apply scoring to every customer in each tier.
    Returns dict: tier -> DataFrame of top_n candidates (descending score).

    Condition assignment:
      Low/High: ranks [0,1]->A  [2,3]->B  [4,5]->C
      Medium:   rank  [0]  ->A  [1]  ->B  [2]  ->C
    """
    COND_LABELS = {
        0: "A -- Blackbox",      1: "A -- Blackbox",
        2: "B -- Standard XAI",  3: "B -- Standard XAI",
        4: "C -- Actionable XAI",5: "C -- Actionable XAI",
    }
    candidates = {}

    for tier, scorer in SCORERS.items():
        sub          = df[df["risk_tier"] == tier].copy()
        sub["score"] = sub.apply(scorer, axis=1)
        ranked       = sub.sort_values("score", ascending=False)
        candidates[tier] = ranked.head(top_n)
        needed       = CANDIDATES_NEEDED[tier]

        print(f"\n{'='*68}")
        print(f"  {tier.upper()} RISK  "
              f"(target p={TIER_CENTRE[tier]:.2f}  need top {needed} of {len(sub):,})")
        print(f"{'='*68}")
        print(f"  {'Rank':<5} {'Assigned to':<24} {'Index':<8} {'Prob':>7}  "
              f"{'Correct':>7}  {'Actual':>7}  {'Tenure':>7}  {'Score':>7}")
        print(f"  {'--'*32}")

        for rank, (idx, row) in enumerate(ranked.head(min(top_n, needed + 2)).iterrows(), 1):
            cond_label  = COND_LABELS.get(rank - 1, "-- reserve")
            marker      = "->" if (rank - 1) < needed else "  "
            correct_str = "Yes" if row["correct"] else "No"
            print(f"  {marker}{rank:<4} {cond_label:<24} {idx:<8} "
                  f"{row['churn_prob']:>7.4f}  "
                  f"{correct_str:>7}  "
                  f"{int(row[TARGET_COL]):>7}  "
                  f"{int(row['Tenure']):>7}  "
                  f"{row['score']:>7.3f}")

    return candidates


# ==============================================================================
# STEP 6 -- Build condition datasets
# ==============================================================================

def build_conditions(candidates, df, shap_df, n_consistent,
                     model=None, num_cols=None, cat_cols=None):
    """
    Build one dataset per condition (5 rows each: 2L + 1M + 2H).
    No customer appears in more than one condition.

    n_consistent SHAP columns are added to B and C -- identical
    column structure ensures conditions differ only in XAI type.
    """
    TIERS = ["Low", "Medium", "High"]
    conditions = {}

    for cond_key, rank_map in COND_RANK_MAP.items():
        rows = []
        for tier in TIERS:
            for rank_idx in rank_map[tier]:
                customer = candidates[tier].iloc[rank_idx]
                idx      = customer.name
                row      = df.loc[idx, DISPLAY_COLS + [TARGET_COL,
                                                        "churn_prob",
                                                        "risk_tier"]].to_dict()
                row["customer_id"] = f"C{idx}"

                # Conditions B + C: add SHAP columns
                if cond_key in ("B-standard-xai", "C-actionable-xai"):
                    shap_row  = shap_df.loc[idx]
                    top_feats = aggregate_shap_to_human(shap_row, n_top=n_consistent)
                    for feat_rank, (feat, val) in enumerate(top_feats, 1):
                        row[f"shap_rank{feat_rank}_feature"]   = feat
                        row[f"shap_rank{feat_rank}_value"]     = round(float(val), 4)
                        row[f"shap_rank{feat_rank}_direction"] = (
                            "increases churn risk" if val > 0 else "decreases churn risk"
                        )

                # Condition C: add retention action columns.
                # The action set comes from the chosen ACTION_STRATEGY:
                #   "A" -> validate-and-filter the hand-written templates,
                #          only keeping ones the model scores ≥1pp lower
                #          than the customer's baseline.
                #   "B" -> generate live counterfactuals with DiCE and use
                #          those (sparsity-relaxed, diversity-prioritised).
                # Both strategies write the same three columns per slot:
                #   action{i}_title, action{i}_rationale, action{i}_changes
                # action{i}_changes is a JSON-encoded {feature: value} dict
                # so the app can apply the exact change without title-
                # keyword matching.
                if cond_key == "C-actionable-xai":
                    selected_actions = build_actions_for_customer(
                        row=row, tier=tier, model=model,
                        num_cols=num_cols, cat_cols=cat_cols,
                        df_population=df, strategy=ACTION_STRATEGY,
                    )
                    for act_rank, act in enumerate(selected_actions[:3], 1):
                        row[f"action{act_rank}_title"]     = act["title"]
                        row[f"action{act_rank}_rationale"] = act["rationale"]
                        row[f"action{act_rank}_changes"]   = json.dumps(act["changes"])

                rows.append(row)

        cond_df = pd.DataFrame(rows).set_index("customer_id")
        cond_df["condition"] = cond_key
        conditions[cond_key] = cond_df

    return conditions


# ==============================================================================
# STEP 7 -- Validate: zero overlap across conditions
# ==============================================================================

def validate_no_overlap(conditions):
    """Assert that no customer ID appears in more than one condition."""
    id_sets   = {k: set(df.index.tolist()) for k, df in conditions.items()}
    cond_keys = list(id_sets.keys())

    print("\n  Overlap validation:")
    all_clean = True
    for i in range(len(cond_keys)):
        for j in range(i + 1, len(cond_keys)):
            a, b    = cond_keys[i], cond_keys[j]
            overlap = id_sets[a] & id_sets[b]
            if overlap:
                print(f"  FAIL  {a} & {b}: {overlap}")
                all_clean = False
            else:
                print(f"  OK    {a} & {b}: no overlap")

    total_unique   = len(id_sets[cond_keys[0]] | id_sets[cond_keys[1]] | id_sets[cond_keys[2]])
    total_expected = sum(len(v) for v in id_sets.values())
    assert all_clean,             "Customer ID overlap detected across conditions!"
    assert total_unique == total_expected, (
        f"Expected {total_expected} unique IDs, found {total_unique}"
    )
    print(f"\n  Total unique customers : {total_unique}  "
          f"({len(conditions)} conditions x {list(TRIAL_LAYOUT.values())} trials)")


# ==============================================================================
# STEP 8 -- Save outputs
# ==============================================================================

def _save_dual(df, stem, out_dir):
    """Save as .parquet (primary) and .csv (companion)."""
    parquet_path = os.path.join(out_dir, f"{stem}-B.parquet")
    csv_path     = os.path.join(out_dir, f"{stem}-B.csv")
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=True)
    df.to_csv(csv_path, index=True)
    p_kb = os.path.getsize(parquet_path) / 1024
    c_kb = os.path.getsize(csv_path) / 1024
    print(f"  {stem}")
    print(f"    parquet : {p_kb:.1f} KB  |  csv : {c_kb:.1f} KB")


def save_all(candidates, conditions, df, out_dir):
    """Save master selection summary and three condition datasets."""
    SUMMARY_COLS = DISPLAY_COLS + [TARGET_COL, "churn_prob", "risk_tier",
                                    "correct", "score"]

    all_rows = []
    for cond_key, rank_map in COND_RANK_MAP.items():
        for tier, ranks in rank_map.items():
            for rank_idx in ranks:
                w   = candidates[tier].iloc[rank_idx]
                row = {col: w[col] for col in SUMMARY_COLS if col in w.index}
                row["customer_id"]        = f"C{w.name}"
                row["assigned_condition"] = cond_key
                row["assigned_tier"]      = tier
                all_rows.append(row)

    summary_df = pd.DataFrame(all_rows).set_index("customer_id")
    _save_dual(summary_df, "XAI-selected-customers-5-B", out_dir)
    print()

    for cond_key, cond_df in conditions.items():
        _save_dual(cond_df, f"cond-{cond_key}", out_dir)
        print()


# ==============================================================================
# STEP 9 -- Selection report figure
# ==============================================================================

def save_selection_report(candidates, path):
    """6-panel diagnostic figure."""
    TIER_COLOR = {"Low": "#3B6D11", "Medium": "#854F0B", "High": "#A32D2D"}
    TIERS      = ["Low", "Medium", "High"]

    fig = plt.figure(figsize=(16, 11), facecolor="#ffffff")
    fig.suptitle("Customer Selection Report -- XAI Experiment (5 Trials per Condition)",
                 fontsize=14, fontweight="bold", y=0.98, color="#1a1a1a")
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35,
                            left=0.07, right=0.97, top=0.92, bottom=0.07)

    # Panels A-C: score distributions
    for col, tier in enumerate(TIERS):
        ax     = fig.add_subplot(gs[0, col])
        ranked = candidates[tier]
        needed = CANDIDATES_NEEDED[tier]
        scores = ranked["score"].values
        colors = [TIER_COLOR[tier] if i < needed else "#dddddd"
                  for i in range(len(scores))]

        ax.barh(range(len(scores))[::-1], scores,
                color=colors, height=0.6, edgecolor="none")

        cond_labels = {0:"->A", 1:"->A", 2:"->B", 3:"->B", 4:"->C", 5:"->C"}
        cond_colors = {0:TIER_COLOR[tier],1:TIER_COLOR[tier],
                       2:"#378ADD",3:"#378ADD",4:"#EF9F27",5:"#EF9F27"}
        for i in range(min(needed, len(scores))):
            ax.text(scores[i] + 0.05, len(scores)-1-i,
                    cond_labels.get(i,""),
                    va="center", fontsize=6.5,
                    color=cond_colors.get(i,"#888"), fontweight="bold")

        ax.set_yticks(range(len(scores))[::-1])
        ax.set_yticklabels([f"C{i}" for i in ranked.index], fontsize=7)
        ax.set_xlabel("Composite score", fontsize=9)
        ax.set_title(f"{tier} risk -- top {len(scores)} candidates",
                     fontsize=10, fontweight="bold", pad=6, color=TIER_COLOR[tier])
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    # Panel D: score component breakdown
    ax4 = fig.add_subplot(gs[1, 0])
    comp_labels = ["Accuracy","Correct\nprediction","Valid label","Feature\nalignment"]
    comp_colors = ["#378ADD","#639922","#1D9E75","#7F77DD"]

    breakdown = {}
    for tier in TIERS:
        w   = candidates[tier].iloc[0]
        tol = 0.20 if tier == "High" else 0.15
        acc = max(0.0, 1.0 - abs(w["churn_prob"] - TIER_CENTRE[tier]) / tol)
        cor = 2.0 if w["correct"] else 0.0
        val = (1.5 if w[TARGET_COL]==1 and tier!="Low"
               else 1.0 if w[TARGET_COL]==0 and tier=="Low" else 0.0)
        feat = max(0.0, w["score"] - acc - cor - val)
        breakdown[tier] = [acc, cor, val, feat]

    x = np.arange(len(TIERS)); bw = 0.55; bottoms = np.zeros(len(TIERS))
    for label, color, vals in zip(comp_labels, comp_colors,
                                  [[breakdown[t][i] for t in TIERS] for i in range(4)]):
        ax4.bar(x, vals, bw, bottom=bottoms, color=color, label=label, alpha=0.85)
        bottoms += np.array(vals)
    ax4.set_xticks(x); ax4.set_xticklabels(TIERS, fontsize=9)
    ax4.set_ylabel("Score points", fontsize=9)
    ax4.set_title("Score breakdown -- Cond A winners", fontsize=10, fontweight="bold", pad=6)
    ax4.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0))
    ax4.spines[["top","right"]].set_visible(False)

    # Panel E: key feature comparison
    ax5 = fig.add_subplot(gs[1, 1])
    feat_avgs = {"Avg Tenure": [], "Avg Charges": [], "Avg Prob (%)": []}
    for cond_key, rank_map in COND_RANK_MAP.items():
        tens  = [candidates[t].iloc[r]["Tenure"] for t,ranks in rank_map.items() for r in ranks]
        chgs  = [candidates[t].iloc[r]["Monthly Charges"] for t,ranks in rank_map.items() for r in ranks]
        probs = [candidates[t].iloc[r]["churn_prob"]*100 for t,ranks in rank_map.items() for r in ranks]
        feat_avgs["Avg Tenure"].append(np.mean(tens))
        feat_avgs["Avg Charges"].append(np.mean(chgs))
        feat_avgs["Avg Prob (%)"].append(np.mean(probs))

    x2 = np.arange(len(feat_avgs)); w2 = 0.25
    cond_clrs = ["#639922","#378ADD","#EF9F27"]
    cond_names = ["A","B","C"]
    for i,(cn,clr) in enumerate(zip(cond_names,cond_clrs)):
        raw   = [feat_avgs[f][i] for f in feat_avgs]
        norms = [v/max(feat_avgs[f]) for f,v in zip(feat_avgs,raw)]
        bars  = ax5.bar(x2+i*w2, norms, w2, color=clr, label=f"Cond {cn}", alpha=0.85)
        for bar,rv in zip(bars,raw):
            ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f"{rv:.0f}", ha="center", fontsize=7)
    ax5.set_xticks(x2+w2); ax5.set_xticklabels(list(feat_avgs.keys()), fontsize=9)
    ax5.set_ylabel("Normalised mean", fontsize=8)
    ax5.set_title("Avg features by condition", fontsize=10, fontweight="bold", pad=6)
    ax5.legend(fontsize=8, frameon=False); ax5.set_ylim(0, 1.3)
    ax5.spines[["top","right"]].set_visible(False)

    # Panel F: prob vs score scatter
    ax6 = fig.add_subplot(gs[1, 2])
    tier_clrs = {"Low":"#639922","Medium":"#EF9F27","High":"#E24B4A"}
    for tier in TIERS:
        ranked = candidates[tier]; needed = CANDIDATES_NEEDED[tier]
        tc     = tier_clrs[tier]
        ax6.scatter(ranked["churn_prob"], ranked["score"],
                    color=tc, alpha=0.25, s=25, label=tier)
        ax6.scatter(ranked.head(needed)["churn_prob"],
                    ranked.head(needed)["score"],
                    color=tc, s=80, zorder=5, edgecolors="#1a1a1a", linewidths=1.0)
        ax6.axvline(TIER_CENTRE[tier], color=tc, lw=0.8, ls="--", alpha=0.5)
    ax6.set_xlabel("Churn probability", fontsize=9)
    ax6.set_ylabel("Composite score", fontsize=9)
    ax6.set_title("Prob vs score\n(solid = assigned to experiment)",
                  fontsize=10, fontweight="bold", pad=6)
    ax6.legend(fontsize=7.5, frameon=False)
    ax6.spines[["top","right"]].set_visible(False)

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    print(f"  Report : {path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("  03_select_and_build_conditions.py")
    print("  IBM Telco Churn -- XAI Experiment Dataset Builder")
    print("="*60)
    print(f"\n  Trial layout   : {TRIAL_LAYOUT}  (per condition)")
    print(f"  Conditions     : {N_CONDITIONS}")
    print(f"  Total customers: {sum(CANDIDATES_NEEDED.values())} unique (no overlap)")
    print(f"  SHAP threshold : {SHAP_COVERAGE_THRESHOLD:.0%}\n")

    print("[ Step 1 ] Loading data and generating predictions ...\n")
    df, model, num_cols, cat_cols = load_and_predict(IN_PARQUET, IN_MODEL)

    print("\n[ Step 2 ] Computing SHAP values (TreeExplainer) ...")
    shap_df, feature_names = compute_shap(model, df, num_cols, cat_cols)

    print("\n[ Step 3 ] Scoring and ranking candidates per tier ...")
    candidates = select_candidates(df, top_n=max(CANDIDATES_NEEDED.values()) + 4)

    print("\n[ Step 4 ] Determining N_CONSISTENT (SHAP coverage threshold) ...")
    # Only B and C customers need coverage evaluation
    bc_indices = [
        candidates[tier].iloc[rank_idx].name
        for cond_key, rank_map in COND_RANK_MAP.items()
        if cond_key in ("B-standard-xai", "C-actionable-xai")
        for tier, ranks in rank_map.items()
        for rank_idx in ranks
    ]
    n_consistent = determine_consistent_n(shap_df, bc_indices,
                                          threshold=SHAP_COVERAGE_THRESHOLD)

    print("\n[ Step 5 ] Building condition datasets ...")
    conditions = build_conditions(candidates, df, shap_df, n_consistent,
                                  model=model, num_cols=num_cols, cat_cols=cat_cols)
    for cond_key, cond_df in conditions.items():
        print(f"  {cond_key:<28} {cond_df.shape[0]} rows x {cond_df.shape[1]} cols  "
              f"tiers={cond_df['risk_tier'].tolist()}")

    print("\n[ Step 6 ] Validating customer uniqueness across conditions ...")
    validate_no_overlap(conditions)

    print("\n[ Step 7 ] Saving outputs (parquet + csv) ...\n")
    save_all(candidates, conditions, df, OUT_DIR)

    print("[ Step 8 ] Saving selection report figure ...")
    save_selection_report(candidates, os.path.join("evaluation", "XAI-selection-report-5-B.png"))

    # Final summary table
    print(f"\n[ Summary ]  N_CONSISTENT = {n_consistent} SHAP features "
          f"(threshold {SHAP_COVERAGE_THRESHOLD:.0%})\n")
    print(f"  {'Condition':<28} {'Tier':<8} {'Customer':<10} "
          f"{'Prob':>7}  {'Actual':>7}  {'Score':>7}  {'Tenure':>7}  Contract")
    print(f"  {'--'*44}")

    for cond_key, rank_map in COND_RANK_MAP.items():
        first = True
        for tier in ["Low", "Medium", "High"]:
            for rank_idx in rank_map[tier]:
                w     = candidates[tier].iloc[rank_idx]
                label = cond_key if first else ""
                print(f"  {label:<28}  {tier:<8}  C{w.name:<9}  "
                      f"{w['churn_prob']:>7.4f}  "
                      f"{int(w[TARGET_COL]):>7}  "
                      f"{w['score']:>7.3f}  "
                      f"{int(w['Tenure']):>7}  {w['Contract']}")
                first = False
        print()

    # ── Sanity check: app-side prediction must equal stored churn_prob ────────
    # Mirrors the apps' predict_risk() exactly. If this assertion ever fails,
    # the displayed risk in the Streamlit dashboards will diverge from the
    # value that drove stimulus selection and that the SHAP values explain.
    print("[ Step 9 ] Verifying app-side prediction matches stored churn_prob ...")
    try:
        eng_stats_path = os.path.join("models", "engineering-stats.joblib")
        eng_stats = joblib.load(eng_stats_path) if os.path.exists(eng_stats_path) else None

        _SVC = ["Phone Service","Multiple Lines","Online Security","Online Backup",
                "Device Protection","Tech Support","Streaming TV","Streaming Movies"]

        def _app_engineer(row_df, q75):
            d = row_df.copy()
            for c in ["Tenure","Monthly Charges","Total Charges"]:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
            t = d["Tenure"].replace(0,1)
            s = [c for c in _SVC if c in d.columns]
            d["Avg_Charge_Per_Month"]   = (d["Total Charges"]/t).round(4)
            d["Services_Count"]         = d[s].apply(lambda r:(r=="Yes").sum(), axis=1)
            d["Charge_Per_Service"]     = (d["Monthly Charges"]/(d["Services_Count"]+1)).round(4)
            d["Is_New_Customer"]        = (d["Tenure"]<=3).astype(int)
            d["Is_Long_Tenured"]        = (d["Tenure"]>=48).astype(int)
            d["Has_Support"]            = ((d["Tech Support"]=="Yes")|(d["Online Security"]=="Yes")).astype(int)
            d["Has_No_Online_Services"] = ((d["Online Security"]=="No")&(d["Online Backup"]=="No")&
                                           (d["Device Protection"]=="No")&(d["Tech Support"]=="No")).astype(int)
            d["Is_Month_To_Month"]      = (d["Contract"]=="Month-to-month").astype(int)
            d["Uses_Paperless_AutoPay"] = ((d["Paperless Billing"]=="Yes")&
                                           (d["Payment Method"]=="Bank transfer (automatic)")).astype(int)
            d["High_Charge_New"]        = ((d["Monthly Charges"]>=q75)&(d["Tenure"]<=12)).astype(int)
            return d

        if eng_stats is None:
            print("  ⚠ engineering-stats.joblib not found -- "
                  "rerun 02-data-processing.py before this script.")
        else:
            q75 = float(eng_stats["monthly_charges_q75"])
            model_cols = list(load_and_predict.__defaults__) if False else None  # noqa
            # Use the same column order the model was trained on:
            num_cols_chk = [c for c in df.select_dtypes(include="number").columns
                            if c not in {TARGET_COL, "churn_prob", "pred_churn", "correct"}]
            cat_cols_chk = [c for c in df.select_dtypes(include="category").columns
                            if c != TARGET_COL]
            expected_cols = num_cols_chk + cat_cols_chk

            max_diff = 0.0
            n_checked = 0
            for cond_key, cond_df in conditions.items():
                for cust_id, row in cond_df.iterrows():
                    one = pd.DataFrame([{c: row[c] for c in DISPLAY_COLS}])
                    one = _app_engineer(one, q75)
                    one = one.reindex(columns=expected_cols, fill_value=0)
                    app_prob = float(model.predict_proba(one)[0][1])
                    stored   = float(row["churn_prob"])
                    max_diff = max(max_diff, abs(app_prob - stored))
                    n_checked += 1
            print(f"  Customers checked  : {n_checked}")
            print(f"  Max |app - stored| : {max_diff:.6f}")
            if max_diff > 1e-4:
                raise AssertionError(
                    f"App-side prediction drifts from stored churn_prob by "
                    f"{max_diff:.6f}. Re-check engineer_features() and the "
                    f"apps' _engineer() / engineering-stats.joblib."
                )
            print("  PASS -- displayed risk will match stimulus-selection probability.")
    except AssertionError:
        raise
    except Exception as _e:
        print(f"  ⚠ Sanity check skipped: {_e}")

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build experiment condition datasets. "
                    "Option A = filtered hand-written actions; "
                    "Option B = live DiCE counterfactuals."
    )
    parser.add_argument("--strategy", choices=["A", "B"],
                        default=ACTION_STRATEGY,
                        help="Which AI-action strategy to use for Condition C. "
                             "Default reads ACTION_STRATEGY env var (or 'A').")
    args = parser.parse_args()
    ACTION_STRATEGY = args.strategy
    print(f"  → ACTION_STRATEGY = {ACTION_STRATEGY}")
    main()