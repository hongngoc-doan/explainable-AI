"""
actionable.py — Condition C: Actionable XAI Dashboard (5-Trial)
===============================================================
Participants see: customer profile + churn probability + SHAP local explanation + AI suggestions
Review screen: shown after last trial, before completion/Unipark redirect.

Run: streamlit run actionable.py
"""

import streamlit as st
st.set_page_config(
    page_title="TelcoX Retention Copilot — System C",
    page_icon="💼", layout="wide",
    initial_sidebar_state="expanded",
)

import os, time, random, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────────────

TOTAL_TRIALS = 5
CONDITION    = "Actionable-XAI"
DATA_FILE    = os.path.join("data", "cond-C-actionable-xai-B.parquet")

RISK_LOW_MAX = 0.30
RISK_MED_MAX = 0.60
MIN_TRIAL_SECONDS = 8

FEATURES_TO_VARY = [
    "Contract", "Tech Support", "Online Security",
    "Payment Method", "Online Backup",
    "Device Protection", "Paperless Billing",
]

SANDBOX_FEATURES = {
    "Contract":          ["Month-to-month", "One year", "Two year"],
    "Internet Service":  ["Fiber optic", "DSL", "No"],
    "Payment Method":    ["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"],
    "Online Security":   ["Yes", "No"],
    "Tech Support":      ["Yes", "No"],
    "Online Backup":     ["Yes", "No"],
    "Device Protection": ["Yes", "No"],
    "Paperless Billing": ["Yes", "No"],
}

MILESTONE_MESSAGES = {
    1: ("First case — read carefully before deciding.", "🚀"),
    2: ("Building momentum — stay systematic.", "💡"),
    3: ("Halfway there. You're doing well!", "🌟"),
    4: ("One more after this — almost done.", "🔥"),
    5: ("Final case! Make it count.", "🎯"),
}

NON_ACTIONABLE = {
    "Tenure (months)", "Total charges ($)", "Avg charge per month ($)",
    "New customer flag", "Long-tenured flag", "High charge + new customer",
    "Number of services", "Cost per service ($)",
}

# Display-name normalisation for SHAP feature labels.
# The experimental data ships the feature label "Avg charge per month ($)";
# expanded here at render time so participants never see the abbreviation.
# Keep the dict tiny and explicit -- partial fuzzy mapping would risk
# silently changing other feature names mid-experiment.
SHAP_DISPLAY_NAMES = {
    "Avg charge per month ($)": "Average charge per month ($)",
}

def _pretty_feature(name: str) -> str:
    return SHAP_DISPLAY_NAMES.get(str(name), str(name))

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
/* ── Force light theme so the dashboard renders consistently regardless of
   the visitor's OS / browser preference (dark vs light). All hardcoded
   light-mode colors below assume a white app background. ─────────────── */
:root,html{color-scheme:light !important;}
html,body,.stApp,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stHeader"],[data-testid="stMain"],section.main,.main{background-color:#ffffff !important;color:#0f172a !important;}
[data-testid="stSidebar"],[data-testid="stSidebar"] > div:first-child{background-color:#f8fafc !important;}
[data-testid="stSidebar"] *{color:#0f172a;}
[data-testid="stMarkdownContainer"]{color:#0f172a;}
[data-testid="stMarkdownContainer"] :where(p,li,h1,h2,h3,h4,h5,h6,span):not([style*="color"]){color:#0f172a;}
label,[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *,[data-testid="stCaptionContainer"],.stCaption{color:#334155 !important;}
.stTextInput input,.stNumberInput input,.stTextArea textarea,[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{background-color:#ffffff !important;color:#0f172a !important;}
[data-baseweb="select"] > div{background-color:#ffffff !important;color:#0f172a !important;}
[data-baseweb="select"] svg{color:#64748b !important;fill:#64748b !important;}
[data-baseweb="popover"],[data-baseweb="popover"] *,[data-baseweb="menu"],[role="listbox"]{background-color:#ffffff !important;color:#0f172a !important;}
[role="option"]{color:#0f172a !important;}
[data-testid="stExpander"]{background-color:#ffffff !important;border-color:#e2e8f0 !important;}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:#334155 !important;}
div.stButton > button{background-color:#ffffff;color:#0f172a;border-color:#e2e8f0;}
hr{border-color:#e2e8f0 !important;}
[role="tooltip"],[data-baseweb="tooltip"]{background-color:#0f172a !important;color:#ffffff !important;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.block-container{padding-top:1.5rem!important;padding-bottom:2.5rem!important;}
.grad{font-weight:700;background:-webkit-linear-gradient(135deg,#0369a1,#7c3aed);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
    padding:10px 14px;box-shadow:0 1px 3px rgba(0,0,0,.04);}
div[data-testid="stMetricValue"]{font-size:1.3rem!important;font-weight:700!important;color:#0f172a!important;}
div[data-testid="stMetricLabel"]{font-size:.76rem!important;color:#64748b!important;font-weight:500!important;}
div[data-testid="stMetricDelta"]{font-size:.82rem!important;font-weight:600!important;}
div[data-testid="stForm"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:20px 24px;}
div.stButton>button:first-child{border-radius:8px;font-weight:600;font-family:'DM Sans',sans-serif;
    transition:all .2s ease;letter-spacing:.01em;}
.streamlit-expanderHeader{font-weight:600!important;color:#334155!important;font-size:.9rem!important;}
.risk-badge{display:inline-block;padding:4px 16px;border-radius:20px;font-weight:700;
    font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;}
.step-dots{display:flex;gap:10px;align-items:center;justify-content:center;padding:8px 0;}
.step-dot{width:12px;height:12px;border-radius:50%;}
.dot-done{background:#0369a1;}
.dot-curr{background:#7c3aed;box-shadow:0 0 0 3px rgba(124,58,237,.22);}
.dot-todo{background:#e2e8f0;}
.stat-pill{background:#f1f5f9;border-radius:8px;padding:6px 10px;font-size:.76rem;
    color:#475569;font-weight:500;display:flex;flex-direction:column;align-items:center;}
.stat-pill strong{font-size:1rem;color:#0f172a;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

for k, v in {
    "logged_in":False,"username":"","customer_count":1,"action_log":[],
    "show_next_ticket":False,"show_review":False,
    "current_customer":None,"ticket_queue":None,"session_seed":None,
    "loaded_for_trial":0,"start_time":0.0,"trial_start_time":0.0,
    "page_interactions":[],"shap_viewed_at":None,"first_action_time":None,
    "sandbox_result":None,"sandbox_sim_count":0,"shap_question":None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Tivian/Unipark integration ────────────────────────────────────────────────
# Reads the participant ID, condition, and return URL from URL query params
# (?pid=...&cond=C&return=...) so the survey can deep-link participants in
# without any manual login. Safe to call on every rerun -- it only runs when
# the relevant params are present.
from unipark_bridge import bootstrap_session_from_url, render_return_button
bootstrap_session_from_url(condition="C")

# ── Artifacts ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model…")
def load_artifacts():
    # Filenames written by 03-models-train.py use hyphens; the apps
    # historically loaded underscored variants. Try both so either layout
    # works without requiring the user to rename anything.
    def _load(*candidates):
        for name in candidates:
            path = os.path.join("models", name)
            if os.path.exists(path):
                return joblib.load(path)
        raise FileNotFoundError(f"None of {candidates} found in models/")
    pipeline      = _load("best-model.joblib")
    expected_cols = _load("model-columns.joblib")
    if "Customer ID" in expected_cols: expected_cols.remove("Customer ID")
    # Engineering stats persisted by 02-data-processing.py. Loading the
    # exact training-time threshold prevents the q75-collapse bug where
    # _engineer() on a 1-row DataFrame computes a per-customer q75.
    try:
        stats = _load("engineering-stats.joblib")
    except FileNotFoundError:
        stats = {"monthly_charges_q75": 89.85}  # IBM Telco fallback
        st.warning("engineering-stats.joblib not found — using fallback q75=89.85. "
                   "Re-run 02-data-processing.py to regenerate.", icon="⚠️")
    return pipeline, expected_cols, stats

@st.cache_data(show_spinner=False)
def load_condition_data():
    df = pd.read_parquet(DATA_FILE)
    if "customer_id" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index":"customer_id"})
    return df

pipeline, expected_cols, eng_stats = load_artifacts()
df_cond = load_condition_data()

if st.session_state.ticket_queue is None and st.session_state.session_seed is not None:
    # Only build the queue AFTER login so each participant gets a different
    # shuffle. Previously the seed defaulted to 42 at module load (before
    # the user had logged in), so every session saw cases in the same order.
    seed = st.session_state.session_seed
    st.session_state.ticket_queue = df_cond.sample(
        frac=1, random_state=seed % (2**31-1)
    ).reset_index(drop=True)

# ── Feature engineering ────────────────────────────────────────────────────────

_SVC = ["Phone Service","Multiple Lines","Online Security","Online Backup",
        "Device Protection","Tech Support","Streaming TV","Streaming Movies"]

def _engineer(df):
    df = df.copy()
    for c in ["Tenure","Monthly Charges","Total Charges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    t = df["Tenure"].replace(0,1)
    s = [c for c in _SVC if c in df.columns]
    df["Avg_Charge_Per_Month"]   = (df["Total Charges"]/t).round(4)
    df["Services_Count"]         = df[s].apply(lambda r:(r=="Yes").sum(),axis=1)
    df["Charge_Per_Service"]     = (df["Monthly Charges"]/(df["Services_Count"]+1)).round(4)
    df["Is_New_Customer"]        = (df["Tenure"]<=3).astype(int)
    df["Is_Long_Tenured"]        = (df["Tenure"]>=48).astype(int)
    df["Has_Support"]            = (
        (df.get("Tech Support",   pd.Series(["No"])).values=="Yes")|
        (df.get("Online Security",pd.Series(["No"])).values=="Yes")).astype(int)
    df["Has_No_Online_Services"] = (
        (df.get("Online Security",  pd.Series(["No"])).values=="No")&
        (df.get("Online Backup",    pd.Series(["No"])).values=="No")&
        (df.get("Device Protection",pd.Series(["No"])).values=="No")&
        (df.get("Tech Support",     pd.Series(["No"])).values=="No")).astype(int)
    df["Is_Month_To_Month"]      = (df.get("Contract",pd.Series([""])).values=="Month-to-month").astype(int)
    df["Uses_Paperless_AutoPay"] = (
        (df.get("Paperless Billing",pd.Series(["No"])).values=="Yes")&
        (df.get("Payment Method",   pd.Series(["" ])).values=="Bank transfer (automatic)")).astype(int)
    # FIX: use the training-time q75 saved by 02-data-processing.py to
    # prevent the q75-collapse bug on 1-row prediction DataFrames.
    q75 = float(eng_stats["monthly_charges_q75"])
    df["High_Charge_New"] = ((df["Monthly Charges"]>=q75)&(df["Tenure"]<=12)).astype(int)
    return df

def predict_risk(customer_dict):
    """
    What-if recomputation only -- the BASELINE risk shown to participants
    is read from the parquet's `churn_prob` column via baseline_risk().
    Returns (probability, engineered_df) to keep app-3's existing call
    signature where the engineered frame is reused for SHAP-related UI.
    """
    skip={"customer_id","Customer ID","Churn","churn_prob","risk_tier","condition"}
    pfx =("shap_rank","action","_")
    clean={k:v for k,v in customer_dict.items()
           if k not in skip and not any(k.startswith(p) for p in pfx)}
    df = pd.DataFrame([clean])
    df = _engineer(df)
    if expected_cols: df=df.reindex(columns=expected_cols,fill_value=0)
    return float(pipeline.predict_proba(df)[0][1]), df

def baseline_risk(customer_dict):
    """
    Canonical churn probability for the customer, taken from the
    stimulus-selection parquet so it stays in lock-step with the SHAP
    values that were computed against it. Falls back to live recomputation
    only if the stored value is somehow missing.
    Returns (probability, engineered_df) -- the engineered frame is
    rebuilt fresh because some downstream UI in app-3 needs it.
    """
    stored = customer_dict.get("churn_prob")
    if stored is not None and pd.notna(stored):
        # Build the engineered frame for downstream UI, but use the stored
        # probability as the displayed value.
        skip={"customer_id","Customer ID","Churn","churn_prob","risk_tier","condition"}
        pfx =("shap_rank","action","_")
        clean={k:v for k,v in customer_dict.items()
               if k not in skip and not any(k.startswith(p) for p in pfx)}
        df = pd.DataFrame([clean])
        df = _engineer(df)
        if expected_cols: df = df.reindex(columns=expected_cols, fill_value=0)
        return float(stored), df
    return predict_risk(customer_dict)

def get_risk_tier(p):
    if p>=RISK_MED_MAX: return "High"
    if p>=RISK_LOW_MAX: return "Medium"
    return "Low"

def get_risk_style(tier):
    return {"High":("HIGH RISK","#dc2626","#fef2f2","#fca5a5"),
            "Medium":("MEDIUM RISK","#d97706","#fffbeb","#fcd34d"),
            "Low":("LOW RISK","#059669","#f0fdf4","#86efac")}[tier]


def _overview_html(cust):
    """Customer-overview cards in a responsive CSS grid.
    Uses minmax(180px, 1fr) so values like "Month-to-month" or
    "Electronic check (automatic)" never truncate; the row wraps
    naturally on narrower viewports."""
    senior = "Yes" if cust.get("Senior Citizen") in [1, "Yes", "1"] else "No"
    cards = [
        ("Tenure",         f"{cust.get('Tenure', 0)} months"),
        ("Monthly Charges", f"${float(cust.get('Monthly Charges', 0)):.2f}"),
        ("Contract",       str(cust.get("Contract", "N/A"))),
        ("Internet Service", str(cust.get("Internet Service", "N/A"))),
        ("Payment Method",  str(cust.get("Payment Method", "N/A"))),
        ("Senior Citizen", senior),
    ]
    cells = "".join(
        f"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:8px;"
        f"padding:10px 14px;min-width:0;'>"
        f"<div style='font-size:.72rem;color:#94a3b8;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;'>{lbl}</div>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#0f172a;"
        f"line-height:1.3;word-wrap:break-word;'>{val}</div></div>"
        for lbl, val in cards
    )
    return ("<div style='display:grid;"
            "grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));"
            "gap:10px;margin-bottom:1.2rem;'>" + cells + "</div>")

# ── Customer loading ───────────────────────────────────────────────────────────

def load_customer():
    q=st.session_state.ticket_queue; idx=st.session_state.customer_count-1
    if q is not None and idx<len(q):
        st.session_state.current_customer =q.iloc[idx].to_dict()
        st.session_state.loaded_for_trial =st.session_state.customer_count
        st.session_state.trial_start_time =time.time()
        st.session_state.start_time       =time.time()
        st.session_state.page_interactions=[]
        st.session_state.shap_viewed_at   =None
        st.session_state.first_action_time=None
        st.session_state.shap_question    =None
        st.session_state.sandbox_result   =None
        st.session_state.sandbox_sim_count=0
    else:
        st.session_state.current_customer=None

def get_safe_index(opts,val):
    try: return opts.index(str(val))
    except: return 0

# ── SHAP from stored values ────────────────────────────────────────────────────

def render_shap_from_stored(customer_row, n=13):
    feats,vals,dirs=[],[],[]
    for i in range(1,n+1):
        f=customer_row.get(f"shap_rank{i}_feature")
        v=customer_row.get(f"shap_rank{i}_value")
        d=customer_row.get(f"shap_rank{i}_direction","")
        if f is None or v is None or str(f)=="nan": break
        feats.append(_pretty_feature(f)); vals.append(float(v)); dirs.append(str(d))
    if not feats:
        fig,ax=plt.subplots(figsize=(7,4),facecolor="none")
        ax.text(0.5,0.5,"SHAP data not available",ha="center",va="center",fontsize=11,color="#94a3b8")
        ax.axis("off"); return fig,"N/A",None
    rf=feats[::-1]; rv=vals[::-1]; rd=dirs[::-1]
    colors=["#ef4444" if "increases" in d.lower() else "#3b82f6" for d in rd]
    fig,ax=plt.subplots(figsize=(7,max(5,len(rf)*.5)),facecolor="none")
    y=list(range(len(rf)))
    ax.barh(y,rv,color=colors,alpha=.82,height=.6,edgecolor="none")
    # Value labels:
    #   Positive bars -> just to the RIGHT of the bar end.
    #   Negative bars -> just to the RIGHT of the zero axis, so the text
    #                    doesn't collide with the y-tick label.
    vmax = max(abs(v) for v in rv) if rv else 1.0
    pad  = vmax * 0.015
    for yi,v in zip(y,rv):
        sign = f"+{v:.3f}" if v >= 0 else f"{v:.3f}"
        if v >= 0:
            ax.text(v + pad, yi, sign, va="center", ha="left",
                    fontsize=7.5, color="#374151", fontweight="500")
        else:
            ax.text(pad, yi, sign, va="center", ha="left",
                    fontsize=7.5, color="#374151", fontweight="500")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{f}  ★" if f.split(":")[0].strip() in NON_ACTIONABLE else f for f in rf],
        fontsize=8.5,color="#334155")
    ax.axvline(0,color="#64748b",linewidth=.9,zorder=3)
    # Extend x-axis so the longest positive label (e.g. "+0.657") doesn't
    # get clipped at the right edge.
    pos_max = max([0] + [v for v in rv if v > 0])
    neg_min = min([0] + [v for v in rv if v < 0])
    ax.set_xlim(neg_min - vmax*0.08, pos_max + vmax*0.12)
    ax.set_xlabel("SHAP value  (impact on churn probability)",fontsize=8,color="#64748b")
    ax.set_title("This Customer's Churn Risk Drivers",pad=14,fontsize=12,fontweight="bold",color="#1e293b")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.tick_params(length=0,colors="#64748b"); ax.grid(axis="x",color="#f1f5f9",linewidth=.8,zorder=0)
    up_p  =mpatches.Patch(color="#ef4444",alpha=.82,label="↑ increases churn risk")
    dn_p  =mpatches.Patch(color="#3b82f6",alpha=.82,label="↓ decreases churn risk")
    note_p=mpatches.Patch(color="none",label="★ = context only, not actionable")
    ax.legend(handles=[up_p,dn_p,note_p],fontsize=7.5,frameon=False,
              loc="lower right",bbox_to_anchor=(1.0,-0.22))
    plt.tight_layout()
    top_feat=feats[0]; top3=feats[:3]
    dists=[f for f in feats if f not in top3]; dist=random.choice(dists) if dists else "Monthly Charges"
    opts=top3+[dist]; random.shuffle(opts)
    shap_q={"question":"According to the AI explanation, which factor had the STRONGEST influence on this customer's churn risk?",
             "options":opts,"answer":top_feat,"correct_idx":opts.index(top_feat)}
    return fig,top_feat,shap_q

# ── Suggestion helpers ─────────────────────────────────────────────────────────

FALLBACK_ACTIONS=[
    {"title":"Switch to automatic payment",
     "rationale":"Customers on bank transfer or credit card auto-pay remove the monthly payment decision — strongly correlated with lower churn."},
    {"title":"Add Online Security + Tech Support",
     "rationale":"Customers using both services churn at rates ~40% lower than those with no sticky add-ons."},
    {"title":"Add Online Backup + Device Protection",
     "rationale":"Data-safety add-ons increase perceived switching costs and product dependency at low cost."},
]

def get_stored_suggestions(row):
    """
    Pull the up-to-3 AI action suggestions from the parquet row.

    Each suggestion may come with three fields:
      - action{i}_title       (str)  -- short headline
      - action{i}_rationale   (str)  -- one-sentence justification
      - action{i}_changes     (str)  -- JSON-encoded {feature: value} dict
                                        with the EXACT changes the model
                                        will apply when this action runs.

    The "changes" column is the new contract (added by script 04 in both
    Option A and Option B). When present it's used directly, bypassing
    the legacy title-keyword mapper. When absent (older parquets) the
    legacy mapper is used as a fallback so the app still functions.
    """
    out=[]
    for i in range(1,4):
        t=row.get(f"action{i}_title"); r=row.get(f"action{i}_rationale")
        c=row.get(f"action{i}_changes")
        ts=str(t) if t and str(t) not in ("nan","None","") else None
        if not ts: continue
        changes=None
        if c and str(c) not in ("nan","None",""):
            try: changes=json.loads(c) if isinstance(c,str) else dict(c)
            except Exception: changes=None
        out.append({"title":ts,"rationale":str(r or ""),"changes":changes})
    fb_iter=iter(FALLBACK_ACTIONS)
    while len(out)<3:
        try:
            fb=next(fb_iter)
            if not any(o["title"]==fb["title"] for o in out):
                out.append({"title":fb["title"],"rationale":fb["rationale"],"changes":None})
        except StopIteration:
            out.append({"title":"Review service plan","rationale":"Discuss the customer's current plan with a retention specialist.","changes":None}); break
    return out[:3]

def map_suggestion_to_changes(title,cust,explicit_changes=None):
    """
    Resolve the post-action customer state.

    If `explicit_changes` is supplied (preferred -- always set when the
    parquet was built by Option A or Option B), apply those changes
    directly. Otherwise fall back to title-keyword inference for backward
    compatibility with parquets built before the changes column existed.
    """
    mod=cust.copy()
    if explicit_changes:
        for k,v in explicit_changes.items():
            mod[k]=v
        return mod
    # ── Legacy title-keyword fallback ──────────────────────────────────
    t=title.lower()
    if "two-year" in t or "two year" in t: mod["Contract"]="Two year"
    elif any(kw in t for kw in ["annual","one-year","one year","contract upgrade","immediate contract","renew"]):
        mod["Contract"]="One year" if cust.get("Contract")=="Month-to-month" else "Two year"
    if "security" in t and "support" in t: mod["Online Security"]="Yes"; mod["Tech Support"]="Yes"
    elif "security" in t: mod["Online Security"]="Yes"
    elif "support" in t: mod["Tech Support"]="Yes"
    if "backup" in t: mod["Online Backup"]="Yes"
    if "device" in t: mod["Device Protection"]="Yes"
    if any(kw in t for kw in ["auto-pay","automatic","payment"]):
        mod["Payment Method"]="Credit card (automatic)"; mod["Paperless Billing"]="Yes"
    if any(kw in t for kw in ["loyalty","outreach","proactive"]):
        if cust.get("Tech Support")!="Yes":  mod["Tech Support"]="Yes"
        if cust.get("Online Backup")!="Yes": mod["Online Backup"]="Yes"
        if cust.get("Payment Method") in ("Electronic check","Mailed check"):
            mod["Payment Method"]="Credit card (automatic)"
    if "maintain" in t:
        if "tech support" in t: mod["Tech Support"]="Yes"
        if "security" in t:     mod["Online Security"]="Yes"
        if "backup" in t:       mod["Online Backup"]="Yes"
    return mod

def get_suggestion_changes_display(title,cust,explicit_changes=None):
    mod=map_suggestion_to_changes(title,cust,explicit_changes)
    check=["Contract","Online Security","Tech Support","Online Backup","Device Protection","Payment Method","Paperless Billing"]
    ch=[f"<b>{c}</b> → <b>{mod[c]}</b>" for c in check if str(cust.get(c,""))!=str(mod.get(c,""))]
    return ch if ch else ["Proactive outreach — no direct service change"]

# ── Action logging ─────────────────────────────────────────────────────────────

def get_changed_features(orig,mod):
    all_f=FEATURES_TO_VARY+list(SANDBOX_FEATURES.keys())
    return [k for k in all_f if str(orig.get(k,""))!=str(mod.get(k,""))]

def calculate_business_cost(details):
    dl=details.lower()
    if "discount" in dl: return "High Cost"
    if any(k in dl for k in ["tech support","security","backup","device"]): return "Medium Cost"
    if any(k in dl for k in ["contract","automatic"]): return "Low Cost"
    return "No Direct Cost"

def handle_action(*, action_type, action_source, details,
                  baseline_prob, modified_cust=None,
                  top_shap_feat=None, used_ai_suggest=False,
                  ai_suggest_idx=None, sandbox_sims=0):
    cust=st.session_state.current_customer
    duration=round(time.time()-st.session_state.start_time,2)
    tier=get_risk_tier(baseline_prob)
    if modified_cust is not None:
        final_prob,_=predict_risk(modified_cust); features_changed=get_changed_features(cust,modified_cust)
    else:
        final_prob=baseline_prob; features_changed=[]
    risk_reduction=baseline_prob-final_prob
    shap_aligned=shap_score=None
    if top_shap_feat and features_changed:
        top_clean=top_shap_feat.split(":")[0].strip().lower()
        matched=[f for f in features_changed if top_clean in f.lower()]
        shap_aligned=len(matched)>0
        shap_score=(1.0 if shap_aligned else
                    0.5 if any(kw in details.lower() for kw in ["security","support","contract","backup","payment"])
                    else 0.0)
    # Mark first action time
    if st.session_state.get("first_action_time") is None:
        st.session_state.first_action_time=round(time.time()-st.session_state.get("trial_start_time",time.time()),2)

    entry={
        "Participant ID":st.session_state.username,"Condition":CONDITION,
        "Trial Number":st.session_state.customer_count,
        "Customer ID":cust.get("customer_id","N/A"),"Risk Tier":tier,
        "Baseline Risk":round(baseline_prob,4),"Modified Risk":round(final_prob,4),
        "Risk Reduction":round(risk_reduction,4),"Risk Reduced":risk_reduction>0.001,
        "Action Type":action_type,"Action Source":action_source,
        "Features Changed":"|".join(features_changed) or "None",
        "Action Details":details,"Business Cost Category":calculate_business_cost(details),
        "Decision Time (sec)":duration,"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Top SHAP Feature":top_shap_feat or "N/A",
        "Action Targets SHAP Driver":shap_aligned,"SHAP Alignment Score":shap_score,
        "Used AI Suggestion":used_ai_suggest,"AI Suggestion Index":ai_suggest_idx,
        "Sandbox Simulations":sandbox_sims,
        "Too Fast Flag":duration<MIN_TRIAL_SECONDS,
        "SHAP View Delay (s)":st.session_state.get("shap_viewed_at"),
        "First Action Time (s)":st.session_state.get("first_action_time"),
        "Interaction Count":len([e for e in st.session_state.get("page_interactions",[])
                                 if e.get("trial")==st.session_state.customer_count]),
        "Self-Reported Confidence":None,"Decision Rationale":None,"SHAP Comprehension Correct":None,
    }
    st.session_state.action_log.append(entry)
    # If the user came from the review screen via "Edit case N", bounce them
    # straight back to the review screen instead of advancing to the next
    # ticket. The new entry has replaced the old one for trial N.
    if st.session_state.get("editing_mode", False):
        st.session_state.editing_mode     = False
        st.session_state.show_review      = True
        st.session_state.show_next_ticket = False
    else:
        st.session_state.show_next_ticket = True
    st.session_state.sandbox_sim_count=0
    st.session_state.shap_viewed_at=None
    st.session_state.first_action_time=None

def get_session_stats():
    log=st.session_state.action_log
    if not log: return {"cases":0,"avg_time":0.0,"actions_taken":0}
    n=len(log); avg=round(sum(e.get("Decision Time (sec)",0) for e in log)/n,1)
    acted=sum(1 for e in log if e.get("Action Type")!="No Action")
    return {"cases":n,"avg_time":avg,"actions_taken":acted}

def estimate_remaining(done,avg_sec):
    rem=TOTAL_TRIALS-done
    if rem<=0: return "done"
    if done==0: return f"~{rem*3} minutes"
    est=rem*avg_sec
    return f"~{int(est)} seconds" if est<120 else f"~{int(est/60)} minutes"

# ══════════════════════════════════════════════════════════════════════════════
# FLOW: LOGIN → (REVIEW) → COMPLETION → POST-TRIAL → [STALE GUARD] → SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

# ── LOGIN ──────────────────────────────────────────────────────────────────────
# Login is handled entirely by bootstrap_session_from_url() near the top of
# the file. That call has three branches:
#   1) Came from Tivian with ?tic=...&pid=...  → auto-login using lfdn as ID,
#      tic stored for the return URL.
#   2) Only one of the two params arrived       → graceful partial login.
#   3) No URL params at all                     → render a manual sign-in
#      form (fallback for dev / pilot / direct access). No return URL is
#      available for manual sessions, so the completion screen falls back
#      to a "task complete, you may close this tab" card.
# By the time execution reaches the lines below, logged_in is True.

# ── REVIEW SCREEN ─────────────────────────────────────────────────────────────
if st.session_state.get("show_review",False):
    # Defensive deduplication: keep the most recent entry per Trial Number.
    # Without this, any flow that produces two entries for the same trial
    # (e.g. an interrupted edit, a double-click on Apply, or a stale session
    # restored from disk) would render two Edit buttons with the same key
    # and trigger Streamlit's DuplicateWidgetID error.
    _seen = {}
    for _e in st.session_state.action_log:
        _tn = _e.get("Trial Number")
        if _tn is not None:
            _seen[_tn] = _e   # later entries overwrite earlier ones
    st.session_state.action_log = list(_seen.values())
    log = st.session_state.action_log

    TC={"High":"#dc2626","Medium":"#d97706","Low":"#059669"}
    TB={"High":"#fef2f2","Medium":"#fffbeb","Low":"#f0fdf4"}
    TD={"High":"#fca5a5","Medium":"#fcd34d","Low":"#86efac"}

    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 .5rem;'>
        <span style='font-size:2.8rem;'>📋</span>
        <h2 style='color:#0f172a;margin:10px 0 4px;font-weight:800;font-size:1.5rem;'>
            Review Your Decisions</h2>
        <p style='color:#64748b;font-size:.95rem;margin:0;'>
            You may edit any case below or confirm and proceed to the post-experiment survey.</p>
    </div>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    # Sort log by trial number so the review list always reads 1..N regardless
    # of edit order.
    sorted_log = sorted(log, key=lambda e: e.get("Trial Number", 0))

    for _row_idx, entry in enumerate(sorted_log):
        baseline=float(entry.get("Baseline Risk",0)); modified=float(entry.get("Modified Risk",baseline))
        delta=modified-baseline; tier_orig=entry.get("Risk Tier","Unknown")
        tier_new=get_risk_tier(modified); action=entry.get("Action Type","—")
        features=entry.get("Features Changed","None"); src_lbl=entry.get("Action Source","")
        trial_num=entry.get("Trial Number","?"); cid=entry.get("Customer ID",f"Case {trial_num}")
        dc="#059669" if delta<-0.001 else ("#dc2626" if delta>0.001 else "#64748b")
        tc_o=TC.get(tier_orig,"#64748b"); tc_n=TC.get(tier_new,"#64748b")
        bg_n=TB.get(tier_new,"#f8fafc"); bd_n=TD.get(tier_new,"#e2e8f0")
        src_badge=("")
        if src_lbl=="AI-Directed": src_badge="<span style='background:#ede9fe;color:#7c3aed;border-radius:4px;padding:1px 7px;font-size:.74rem;font-weight:600;margin-right:4px;'>✨ AI</span>"
        elif src_lbl=="User-Initiated": src_badge="<span style='background:#e0f2fe;color:#0369a1;border-radius:4px;padding:1px 7px;font-size:.74rem;font-weight:600;margin-right:4px;'>👤 Own</span>"
        feat_html=(" · ".join(f"<span style='background:#f1f5f9;color:#334155;border-radius:4px;padding:1px 6px;font-size:.78rem;'>{f.strip()}</span>" for f in features.split("|"))
                   if features and features!="None"
                   else "<span style='color:#94a3b8;font-size:.82rem;'>No service changes</span>")
        arrow="=" if abs(delta)<0.001 else ("↓" if delta<0 else "↑")
        # Card content + per-case Edit button
        col_card, col_edit = st.columns([6, 1])
        with col_card:
            st.markdown(f"""
            <div style='background:#fff;border:1px solid {bd_n};border-radius:12px;padding:14px 18px;margin-bottom:8px;box-shadow:0 1px 4px rgba(0,0,0,.04);'>
                <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
                    <div><span style='font-size:.7rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:.05em;'>Case {trial_num} &middot; {cid}</span><br>
                        <span style='font-size:.94rem;font-weight:700;color:#1e293b;'>{src_badge}{action}</span></div>
                    <div style='display:flex;gap:12px;align-items:center;'>
                        <div style='text-align:center;'>
                            <p style='margin:0;font-size:.65rem;color:#94a3b8;text-transform:uppercase;font-weight:600;'>Before</p>
                            <p style='margin:2px 0 0;font-size:.95rem;font-weight:700;color:{tc_o};'>{baseline*100:.1f}%&nbsp;<span style='font-size:.72rem;font-weight:500;'>{tier_orig}</span></p></div>
                        <span style='font-size:1.4rem;color:{dc};font-weight:700;'>{arrow}&nbsp;<span style='font-size:.85rem;'>{delta*100:+.1f}pp</span></span>
                        <div style='background:{bg_n};border-radius:8px;padding:4px 10px;text-align:center;'>
                            <p style='margin:0;font-size:.65rem;color:#94a3b8;text-transform:uppercase;font-weight:600;'>After</p>
                            <p style='margin:2px 0 0;font-size:.95rem;font-weight:700;color:{tc_n};'>{modified*100:.1f}%&nbsp;<span style='font-size:.72rem;font-weight:500;'>{tier_new}</span></p></div></div></div>
                <div style='margin-top:8px;padding-top:8px;border-top:1px solid #f1f5f9;'>
                    <span style='font-size:.76rem;color:#64748b;font-weight:500;'>Features changed: </span>{feat_html}</div></div>""",
                        unsafe_allow_html=True)
        with col_edit:
            # Edit-this-case button. Removes only the entry for this trial,
            # jumps the queue back to that trial, and flags editing_mode so
            # handle_action returns the user to the review screen instead of
            # advancing to the next case.
            if st.button("✏️ Edit", key=f"edit_case_{trial_num}_{_row_idx}",
                         use_container_width=True):
                st.session_state.action_log = [
                    e for e in st.session_state.action_log
                    if e.get("Trial Number") != trial_num
                ]
                st.session_state.customer_count   = int(trial_num)
                st.session_state.editing_mode     = True
                st.session_state.show_review      = False
                st.session_state.show_next_ticket = False
                st.session_state.trial_start_time = time.time()
                st.session_state.start_time       = time.time()
                st.session_state.loaded_for_trial = 0   # force reload
                load_customer()
                st.rerun()

    st.markdown("<br>",unsafe_allow_html=True)
    col_back,col_confirm=st.columns([1,2])
    with col_back:
        # Unified across A/B/C: the review page no longer has a dedicated
        # "back to last case" shortcut — participants edit any case via the
        # ✏️ Edit buttons next to the case cards. This keeps the review
        # surface identical across conditions so any condition effect on
        # editing behaviour cannot be attributed to extra affordances in
        # the Actionable-XAI arm.
        st.caption("Or close the review and submit:")
    with col_confirm:
        st.markdown("<p style='color:#64748b;font-size:.85rem;margin:0 0 6px;'>Click <b>Confirm & Submit</b> to submit your decisions.</p>",unsafe_allow_html=True)
        if st.button("✅ Confirm & Submit",type="primary",use_container_width=True,key="review_confirm"):
            # Force completion regardless of where customer_count happens to
            # sit. After an edit-then-confirm sequence, customer_count is at
            # the edited trial (e.g. 4), so without this the completion gate
            # `customer_count > TOTAL_TRIALS` never fires and the user gets
            # bounced back to case 4 to redo cases 4..N.
            st.session_state.show_review      = False
            st.session_state.show_review_done = True
            st.session_state.customer_count   = TOTAL_TRIALS + 1
            st.session_state.show_next_ticket = False
            st.session_state.editing_mode     = False
            st.rerun()
    st.stop()

# ── COMPLETION ─────────────────────────────────────────────────────────────────
if st.session_state.customer_count>TOTAL_TRIALS:
    # First hit after last trial → show review
    if not st.session_state.get("show_review_done",False):
        st.session_state.show_review=True
        st.rerun()
    # After review confirmed → save and complete
    os.makedirs("results",exist_ok=True)
    df_log=pd.DataFrame(st.session_state.action_log)
    df_log.to_csv(os.path.join("results",f"{st.session_state.username}-C.csv"),index=False)
    stats=get_session_stats()
    st.markdown("""<div style='text-align:center;margin-top:14vh;'>
        <span style='font-size:4rem;'>✅</span>
        <h1 style='color:#059669;margin:12px 0 4px;font-weight:700;'>Task Complete!</h1>
        <p style='color:#64748b;font-size:1.05rem;'>All decisions recorded. Thank you!</p>
    </div>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Cases Reviewed",TOTAL_TRIALS); c2.metric("Actions Taken",stats["actions_taken"])
    c3.metric("Risk Reductions",sum(1 for e in df_log.to_dict("records") if e.get("Risk Reduced")))
    c4.metric("Average Time",f"{stats['avg_time']} seconds")
    st.markdown("<br>",unsafe_allow_html=True)
    # Returns the participant to Unipark. Falls back to the original
    # 'close this tab' instruction if no return URL was supplied.
    render_return_button()
    st.stop()

# ── POST-TRIAL ─────────────────────────────────────────────────────────────────
if st.session_state.show_next_ticket:
    last=st.session_state.action_log[-1] if st.session_state.action_log else {}
    fr=float(last.get("Modified Risk",0)); bl=float(last.get("Baseline Risk",0))
    delta=fr-bl; dc="#059669" if delta<0 else ("#dc2626" if delta>0 else "#64748b")
    ds=f"{delta*100:+.1f}pp"; dw="reduced" if delta<0 else ("unchanged" if abs(delta)<0.001 else "increased")
    trial=st.session_state.customer_count
    emoji,milestone_msg=MILESTONE_MESSAGES.get(trial,("✅","Case reviewed."))
    tl,tc,_,_=get_risk_style(get_risk_tier(fr))
    st.markdown(f"""<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:28px;text-align:center;max-width:600px;margin:2rem auto;'>
        <span style='font-size:2.5rem;'>{emoji}</span>
        <h2 style='color:#0f172a;margin:10px 0 4px;font-size:1.35rem;font-weight:700;'>Case {trial} Complete</h2>
        <p style='color:#64748b;margin:0 0 18px;font-size:.92rem;'>{milestone_msg}</p>
        <div style='display:flex;gap:14px;justify-content:center;flex-wrap:wrap;'>
            <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 20px;min-width:140px;'>
                <p style='margin:0;font-size:.7rem;color:#94a3b8;font-weight:600;text-transform:uppercase;'>Risk Change</p>
                <p style='margin:4px 0 0;font-size:1.5rem;font-weight:800;color:{dc};'>{ds}</p>
                <p style='margin:0;font-size:.8rem;color:{dc};'>{dw}</p></div>
            <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 20px;min-width:140px;'>
                <p style='margin:0;font-size:.7rem;color:#94a3b8;font-weight:600;text-transform:uppercase;'>New Risk</p>
                <p style='margin:4px 0 0;font-size:1.1rem;font-weight:700;color:{tc};'>{fr*100:.1f}%</p>
                <p style='margin:0;font-size:.8rem;color:{tc};'>{tl}</p></div></div></div>""",unsafe_allow_html=True)

    is_last=trial>=TOTAL_TRIALS
    fin_note=("✅ All cases done — click below to review your decisions before submitting."
              if is_last else "Decision recorded. Proceed to the next customer case.")
    st.markdown(f"<p style='text-align:center;color:#64748b;font-size:.9rem;'>{fin_note}</p>",unsafe_allow_html=True)
    _,btn_col,_=st.columns([1,2,1])
    with btn_col:
        next_lbl="📋 Review and Submit" if is_last else "📄 Continue to Next Case"
        if st.button(next_lbl,type="primary",use_container_width=True,key=f"next_{trial}"):
            st.session_state.customer_count+=1
            st.session_state.show_next_ticket=False
            st.session_state.shap_question=None
            st.session_state.sandbox_result=None
            st.session_state.sandbox_sim_count=0
            if is_last:
                st.session_state.show_review=True
            else:
                load_customer()
            st.rerun()
    st.stop()

# ── STALE-CUSTOMER GUARD ───────────────────────────────────────────────────────
if (st.session_state.current_customer is not None and
        st.session_state.get("loaded_for_trial",0)!=st.session_state.customer_count):
    load_customer(); st.rerun()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    trial=st.session_state.customer_count; cust_s=st.session_state.current_customer
    stats=get_session_stats(); done=stats["cases"]
    st.markdown(f"""<div style='text-align:center;padding:12px 0 6px;'>
        <span style='font-size:2rem;'>🏢</span>
        <p style='margin:6px 0 0;font-size:.68rem;color:#0284c7;font-weight:700;letter-spacing:.18em;text-transform:uppercase;'>TelcoX Telecom</p><h2 style='margin:2px 0 2px;font-size:1.1rem;font-weight:700;color:#0f172a;'>Retention Copilot</h2>
        <span style='font-size:.76rem;color:#64748b;'>Progress</span>
    </div>
    <p style='text-align:center;color:#94a3b8;font-size:.78rem;margin:0 0 8px;'>
        Participant <b style='color:#475569;'>{st.session_state.username}</b></p>""",unsafe_allow_html=True)
    if cust_s is not None:
        dots="<div class='step-dots'>"+"".join(
            f"<div class='step-dot {'dot-done' if i<trial else 'dot-curr' if i==trial else 'dot-todo'}' title='Case {i}'></div>"
            for i in range(1,TOTAL_TRIALS+1))+"</div>"
        mt=MILESTONE_MESSAGES.get(trial,("",""))[0]
        st.markdown(f"""{dots}<p style='text-align:center;color:#64748b;font-size:.78rem;margin:2px 0 8px;'>
            <b>Case {trial} of {TOTAL_TRIALS}</b> · {mt}</p>""",unsafe_allow_html=True)
        st.progress(min((trial-1)/TOTAL_TRIALS,1.0))
        if done>0:
            # Show only the decisions-made indicator. The avg-decision-time
            # and time-remaining pills were removed because a visible
            # countdown can induce time pressure and may bias trust /
            # workload self-reports (Condition C's sandbox naturally
            # takes longer per case, so the "left" pill would show
            # condition-correlated numbers). Decision Time (sec) is
            # still recorded per trial in handle_action() and exported
            # to the CSV.
            st.markdown(f"""<div style='display:flex;gap:5px;margin:10px 0;'>
                <div class='stat-pill' style='flex:1'><strong>{stats["actions_taken"]}/{TOTAL_TRIALS}</strong>acted</div>
            </div>""",unsafe_allow_html=True)
        st.markdown("<hr style='margin:10px 0;border-color:#f1f5f9;'>",unsafe_allow_html=True)
        tid=str(cust_s.get("customer_id","000000"))
        st.markdown(f"""<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:12px;text-align:center;margin:4px 0 10px;'>
            <p style='margin:0;color:#0284c7;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;'>Active Ticket</p>
            <h2 style='margin:2px 0 0;color:#0369a1;font-size:1.5rem;font-weight:800;font-family:"DM Mono",monospace;'>{tid}</h2>
        </div>""",unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:10px;text-align:center;margin:8px 0 6px;'>
            <p style='margin:0;color:#059669;font-size:.82rem;font-weight:600;'>✅ Ready — press Begin to start</p>
        </div><hr style='margin:6px 0 10px;border-color:#f1f5f9;'>""",unsafe_allow_html=True)
    with st.expander("📖 Terminology Guide",expanded=False):
        st.markdown("""<div style='font-size:.82rem;color:#475569;line-height:1.65;'>
        <h5 style='color:#0f172a;margin:6px 0 8px;'>📊 Risk Tiers</h5>
        <b>Churn Probability:</b> The AI's estimated likelihood (0–100%) that a customer will cancel.<br><br>
        <span style='color:#059669;font-weight:600;'>LOW (0–30%)</span> — Stable.<br>
        <span style='color:#d97706;font-weight:600;'>MEDIUM (30–60%)</span> — Elevated.<br>
        <span style='color:#dc2626;font-weight:600;'>HIGH (60%+)</span> — Critical.<br><br>
        <b>Percentage points (pp):</b> The unit used when comparing two probabilities or percentages. A drop from 60% to 45% is a reduction of <b>15 pp</b>, not 15%. Risk changes after an action are shown in pp.
        <hr style='margin:10px 0;border-color:#e2e8f0;'>
        <h5 style='color:#0f172a;margin:6px 0 8px;'>📡 Key Features</h5>
        <b>Tenure:</b> How many months the customer has been with the company. Short tenure = higher risk; long-tenured customers are far more loyal.<br><br>
        <b>Contract:</b> Length of the customer's commitment.
        <ul style='margin:4px 0 0 0;padding-left:20px;'>
            <li><i>Month-to-month</i> — Top churn predictor.</li>
            <li><i>One year</i> — Moderate retention.</li>
            <li><i>Two year</i> — Strongest retention.</li>
        </ul><br>
        <b>Internet Service:</b> Type of internet plan.
        <ul style='margin:4px 0 0 0;padding-left:20px;'>
            <li><i>DSL</i> — Lower price, lower churn signal.</li>
            <li><i>Fiber optic</i> — Higher price + alternatives = moderate churn signal.</li>
            <li><i>No</i> — Customer has no internet service.</li>
        </ul><br>
        <b>Payment Method:</b> How the customer pays.
        <ul style='margin:4px 0 0 0;padding-left:20px;'>
            <li><i>Electronic check</i> — Strongly correlated with churn.</li>
            <li><i>Mailed check</i> — Moderate correlation.</li>
            <li><i>Bank transfer (automatic)</i> / <i>Credit card (automatic)</i> — Autopay reduces churn.</li>
        </ul><br>
        <b>Add-ons</b> (Online Security, Tech Support, Online Backup, Device Protection): Protective factors. Adding = high retention value.<br><br>
        <b>Paperless Billing:</b> Whether the customer has switched to paperless statements.<br><br>
        <b>Senior Citizen:</b> Whether the customer is aged 65+.
        <hr style='margin:10px 0;border-color:#e2e8f0;'>
        <h5 style='color:#0f172a;margin:6px 0 8px;'>🤖 AI Explanation Terms</h5>
        <b>SHAP:</b> 🔴 Red = increases churn risk. 🔵 Blue = decreases it. ★ = context only, not actionable.<br><br>
        <b>Risk if Applied:</b> Model's predicted risk assuming the action is accepted by the customer.
        <hr style='margin:10px 0;border-color:#e2e8f0;'>
        <h5 style='color:#0f172a;margin:6px 0 8px;'>🎯 Actions</h5>
        <b>AI Action:</b> Apply one of the AI-recommended interventions targeting top risk drivers.<br>
        <b>Sandbox:</b> Build your own intervention by adjusting any features in the live what-if simulator.<br>
        <b>No Action:</b> Appropriate when risk is low or no offer is suitable.
        </div>""",unsafe_allow_html=True)

# ── LANDING ────────────────────────────────────────────────────────────────────
if st.session_state.current_customer is None:
    st.markdown("""<h1 class='grad' style='margin-top:-.8rem;'>Welcome to the TelcoX Retention Copilot</h1>
    """,unsafe_allow_html=True)
    st.divider()
    bc1,bc2=st.columns([2,1])
    with bc1:
        st.markdown("""<div style='background:linear-gradient(135deg,#f0f9ff,#ede9fe);border:1px solid #bae6fd;border-radius:10px;padding:12px 16px;margin:0 0 14px;'><p style='margin:0;color:#0f172a;font-size:.92rem;line-height:1.55;'><b>Scenario:</b> You are stepping into the role of a <b>Retention Manager</b> at <b>TelcoX</b>, a fictional telecommunications company. Your team uses an AI-assisted dashboard to flag at-risk customers and decide on retention actions.</p></div>""",unsafe_allow_html=True)
        st.markdown("### 📋 Daily Briefing")
        st.info("**📢 Promotion:** Free Fiber Optic upgrade for customers switching to an annual contract.",icon="📢")
        st.success("**Policy:** Review the AI explanation and AI-recommended actions, then apply the most appropriate retention intervention.",icon="📋")
        st.markdown(f"""<div style='background:#f8fafc;border-left:4px solid #7c3aed;border-radius:0 8px 8px 0;padding:14px 18px;margin-top:12px;'>
            <b style='color:#1e293b;'>Your task — {TOTAL_TRIALS} cases:</b>
            <ol style='margin:8px 0 0;color:#475569;font-size:.92rem;line-height:1.9;'>
                <li>Review the customer profile, risk score, and SHAP explanation.</li>
                <li>Examine the AI-recommended actions with projected risk outcomes.</li>
                <li>Apply an AI action, or test your own approach in the Sandbox.</li>
                <li>After all cases, review and confirm your decisions before submitting.</li>
            </ol></div>""",unsafe_allow_html=True)
        # Helpful pointer to the sidebar guide so participants know it exists.
        st.markdown("""<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;margin-top:10px;display:flex;align-items:flex-start;gap:10px;'>
            <span style='font-size:1.05rem;line-height:1.2;'>💡</span>
            <div style='font-size:.85rem;color:#1e3a8a;line-height:1.55;'>
                <b>Tip:</b> Open the <b>📖 Terminology Guide</b> in the left sidebar at any time to look up what each field, risk tier, or action means.
            </div></div>""",unsafe_allow_html=True)
    with bc2:
        st.markdown("### 📊 Your Session")
        st.markdown(f"""<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:18px;'>
            <p style='margin:0 0 10px;color:#0f172a;font-weight:500;'>👤 <b>Participant:</b> {st.session_state.username}</p>
            <p style='margin:0 0 10px;color:#0f172a;font-weight:500;'>📋 <b>Cases:</b> {TOTAL_TRIALS}</p>
            <p style='margin:0;color:#0f172a;font-weight:500;'>⏱️ <b>Estimated time:</b> 10–15 minutes</p>
        </div>""",unsafe_allow_html=True)
    # Reminder before the button: the landing/introduction page cannot be
    # reached again once the first case is loaded, so participants should
    # finish reading the briefing before continuing.
    st.markdown("""<div style='background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;margin:1.4rem 0 0;display:flex;align-items:flex-start;gap:10px;'>
        <span style='font-size:1.05rem;line-height:1.2;'>⚠️</span>
        <div style='font-size:.88rem;color:#78350f;line-height:1.55;'>
            <b>Please note:</b> Once you click <b>Begin First Case</b>, you will not be able to return to this introduction page. Please make sure you have read the briefing above before continuing.
        </div></div>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("📄 Begin First Case",type="primary",use_container_width=True):
        load_customer(); st.rerun()
    st.stop()

# ── MAIN DASHBOARD ─────────────────────────────────────────────────────────────
cust=st.session_state.current_customer
display_id=cust.get("customer_id","Unknown")
baseline_prob,df_instance=baseline_risk(cust)
tier=get_risk_tier(baseline_prob)
label,text_col,bg_col,border_col=get_risk_style(tier)
top_shap_feat=None

st.markdown("<h3 class='grad' style='margin-top:-.5rem;'>📋 Customer Telemetry</h3>",unsafe_allow_html=True)
st.markdown(f"""<div style='background:{bg_col};border:1.5px solid {border_col};border-radius:18px;padding:22px;text-align:center;max-width:520px;margin:0 auto 1.5rem;'>
    <p style='margin:0;color:#64748b;font-weight:600;font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;'>Churn Probability</p>
    <h1 style='margin:6px 0 4px;font-size:3.8rem;font-weight:800;color:{text_col};line-height:1;'>{baseline_prob*100:.1f}%</h1>
    <span class='risk-badge' style='background:{border_col}22;border:1px solid {border_col};color:{text_col};'>{label}</span>
</div>""",unsafe_allow_html=True)

st.markdown("##### 👤 Customer Overview")
st.markdown(_overview_html(cust), unsafe_allow_html=True)

with st.expander("📂 Full Customer Record",expanded=False):
    c1,c2,c3,c4=st.columns(4)
    for col,pairs in {c1:[("Partner","Partner"),("Dependents","Dependents"),("Phone Service","Phone Service")],
                      c2:[("Multiple Lines","Multiple Lines"),("Online Security","Online Security"),("Online Backup","Online Backup")],
                      c3:[("Device Protection","Device Protection"),("Tech Support","Tech Support"),("Streaming TV","Streaming TV")],
                      c4:[("Streaming Movies","Streaming Movies"),("Paperless Billing","Paperless Billing"),("Total Charges","Total Charges")]}.items():
        with col:
            for key,lbl in pairs:
                val=cust.get(key,"N/A")
                if key=="Total Charges" and val!="N/A":
                    try: val=f"${float(val):.2f}"
                    except: pass
                st.write(f"**{lbl}:** {val}")

# ── SHAP ───────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🔍 AI Explanation (SHAP)")
st.markdown("<p style='color:#64748b;font-size:.88rem;margin-top:-8px;'>🔴 Red = increases churn risk. 🔵 Blue = decreases it. ★ = context only, not directly actionable by retention specialists.</p>",unsafe_allow_html=True)
try:
    shap_fig,top_feat,shap_q=render_shap_from_stored(cust)
    if st.session_state.shap_question is None and shap_q: st.session_state.shap_question=shap_q
    top_shap_feat=top_feat
    if st.session_state.get("shap_viewed_at") is None:
        st.session_state.shap_viewed_at=round(time.time()-st.session_state.get("trial_start_time",time.time()),2)
    _,chart_col,_=st.columns([0.5,9,0.5])
    with chart_col: st.pyplot(shap_fig,use_container_width=True)
    if top_feat and top_feat!="N/A":
        clean_top=top_feat.split(":")[0].strip()
        st.markdown(f"""<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:10px 16px;margin-top:6px;'>
            <p style='margin:0;color:#0369a1;font-size:.85rem;'><b>🔑 Top driver:</b> <b>{clean_top}</b> — the AI recommendations below address this and other high-impact drivers.</p>
        </div>""",unsafe_allow_html=True)
except Exception as e:
    st.error(f"SHAP chart error: {e}")

# ── AI RECOMMENDATIONS ─────────────────────────────────────────────────────────
st.divider()
st.markdown("<h3 class='grad'>✨ AI-Recommended Retention Actions</h3>",unsafe_allow_html=True)
desc=("Targeted interventions generated from the SHAP analysis for this customer. Each directly addresses one of the top risk drivers, with the projected risk outcome shown below each card."
      if baseline_prob>=0.30 else "AI-suggested actions to reinforce this customer's loyalty and increase their long-term value.")
st.markdown(f"<p style='color:#64748b;font-size:.9rem;margin-top:-6px;'><i>{desc}</i></p>",unsafe_allow_html=True)

# Pre-compute BEFORE rendering columns — ensures equal card heights
all_sugs=get_stored_suggestions(cust)
BADGE_COLS=["#0369a1","#7c3aed","#059669"]
sug_data=[]
for sug in all_sugs:
    mod_cust=map_suggestion_to_changes(sug["title"],cust,sug.get("changes"))
    proj_prob,_=predict_risk(mod_cust)
    delta=proj_prob-baseline_prob
    chg_list=get_suggestion_changes_display(sug["title"],cust,sug.get("changes"))
    sug_data.append({"sug":sug,"mod":mod_cust,"prob":proj_prob,"delta":delta,"changes":chg_list})

sug_cols=st.columns(len(sug_data))

for i,d in enumerate(sug_data):
    sug=d["sug"]; mod_cust=d["mod"]; proj_prob=d["prob"]; delta=d["delta"]; chg_list=d["changes"]
    bc=BADGE_COLS[i%len(BADGE_COLS)]
    chg_html="".join(f"<p style='margin:4px 0;color:#334155;font-size:.84rem;'>🔹 {c}</p>" for c in chg_list)
    cap_icon,cap_txt=(("✅","Reduces risk for this customer.") if delta<-0.001
                      else ("ℹ️","Minimal risk impact — consider the Sandbox.") if abs(delta)<=0.001
                      else ("⚠️","This action may not reduce risk for this customer profile."))
    with sug_cols[i]:
        st.markdown(f"""<div style='background:#fff;border:1.5px solid {bc}33;border-radius:12px;
                    padding:16px 18px;box-shadow:0 2px 6px rgba(0,0,0,.04);
                    height:260px;overflow-y:auto;display:flex;flex-direction:column;'>
            <p style='margin:0 0 4px;font-size:.7rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:.05em;'>AI Action {i+1}</p>
            <h4 style='color:{bc};margin:0 0 8px;font-weight:700;font-size:.95rem;'>✨ {sug["title"]}</h4>
            <p style='margin:0 0 10px;color:#64748b;font-size:.82rem;line-height:1.55;flex:1;'>{sug["rationale"]}</p>
            <div style='border-top:1px solid #f1f5f9;padding-top:8px;'>{chg_html}</div>
        </div>""",unsafe_allow_html=True)
        st.metric("Risk if applied",f"{proj_prob*100:.1f}%",f"{delta*100:+.1f}%",delta_color="inverse")
        st.caption(f"{cap_icon} {cap_txt}")
        if st.button(f"Apply",key=f"sug_{i}_{display_id}",use_container_width=True,type="secondary"):
            handle_action(action_type=f"AI Recommendation {i+1}",action_source="AI-Directed",
                details=f"{sug['title']} | {sug['rationale']}",baseline_prob=baseline_prob,
                modified_cust=mod_cust,top_shap_feat=top_shap_feat,
                used_ai_suggest=True,ai_suggest_idx=i+1)
            st.rerun()

# ── SCENARIO SANDBOX ──────────────────────────────────────────────────────────
st.divider()
st.markdown("<h3 style='color:#334155;margin-bottom:2px;'>🧪 Scenario Sandbox</h3>",unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;font-size:.9rem;margin-top:0;'>Simulate any custom configuration and see the projected churn probability. Use this when no AI action fits.</p>",unsafe_allow_html=True)

with st.form(f"sandbox_{display_id}"):
    st.caption("Adjust any combination — only changed features will be applied.")
    fc1=st.columns(4)
    new_contract=fc1[0].selectbox("Contract",SANDBOX_FEATURES["Contract"],index=get_safe_index(SANDBOX_FEATURES["Contract"],cust.get("Contract")))
    new_internet=fc1[1].selectbox("Internet Service",SANDBOX_FEATURES["Internet Service"],index=get_safe_index(SANDBOX_FEATURES["Internet Service"],cust.get("Internet Service")))
    new_payment =fc1[2].selectbox("Payment Method",SANDBOX_FEATURES["Payment Method"],index=get_safe_index(SANDBOX_FEATURES["Payment Method"],cust.get("Payment Method")))
    new_paper   =fc1[3].selectbox("Paperless Billing",SANDBOX_FEATURES["Paperless Billing"],index=get_safe_index(SANDBOX_FEATURES["Paperless Billing"],cust.get("Paperless Billing")))
    fc2=st.columns(4)
    new_sec   =fc2[0].selectbox("Online Security",SANDBOX_FEATURES["Online Security"],index=get_safe_index(SANDBOX_FEATURES["Online Security"],cust.get("Online Security")))
    new_tech  =fc2[1].selectbox("Tech Support",SANDBOX_FEATURES["Tech Support"],index=get_safe_index(SANDBOX_FEATURES["Tech Support"],cust.get("Tech Support")))
    new_backup=fc2[2].selectbox("Online Backup",SANDBOX_FEATURES["Online Backup"],index=get_safe_index(SANDBOX_FEATURES["Online Backup"],cust.get("Online Backup")))
    new_device=fc2[3].selectbox("Device Protection",SANDBOX_FEATURES["Device Protection"],index=get_safe_index(SANDBOX_FEATURES["Device Protection"],cust.get("Device Protection")))
    st.markdown("<br>",unsafe_allow_html=True)
    simulate=st.form_submit_button("⚡ Simulate Risk Impact",type="primary")

if simulate:
    sim=cust.copy()
    sim.update({"Contract":new_contract,"Internet Service":new_internet,"Payment Method":new_payment,
                "Paperless Billing":new_paper,"Online Security":new_sec,"Tech Support":new_tech,
                "Online Backup":new_backup,"Device Protection":new_device})
    sim_prob,_=predict_risk(sim)
    changed=any(str(sim.get(k,""))!=str(cust.get(k,"")) for k in SANDBOX_FEATURES)
    st.session_state.sandbox_result={"prob":sim_prob,"delta":sim_prob-baseline_prob,
        "changed":changed,"cust_id":display_id,
        "details":(f"Contract:{new_contract}|Internet:{new_internet}|Pay:{new_payment}|"
                   f"Paperless:{new_paper}|Sec:{new_sec}|Tech:{new_tech}|Backup:{new_backup}|Device:{new_device}"),
        "cust":sim}
    st.session_state.sandbox_sim_count+=1

res=st.session_state.get("sandbox_result")
if res and res.get("cust_id")==display_id:
    n_sims=st.session_state.sandbox_sim_count
    dc="#059669" if res["delta"]<0 else "#dc2626"
    rt=get_risk_tier(res["prob"]); _,rtc,_,_=get_risk_style(rt)
    rc1,rc2,rc3=st.columns([3,2,1])
    with rc1:
        st.markdown(f"""<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;'>
            <p style='margin:0 0 4px;color:#64748b;font-size:.8rem;font-weight:500;'>Simulation {n_sims} — projected risk</p>
            <p style='margin:0;font-size:1.05rem;color:#0f172a;'><b style='color:{rtc};'>{res["prob"]*100:.1f}%</b>
            &nbsp;<span style='color:{dc};font-weight:700;'>({res["delta"]*100:+.1f}pp)</span></p>
        </div>""",unsafe_allow_html=True)
    with rc2:
        if res["changed"]:
            if st.button("✅ Apply Configuration",type="primary",use_container_width=True,key=f"apply_sb_{display_id}_{n_sims}"):
                handle_action(action_type="Sandbox Custom Change",action_source="User-Initiated",
                    details=res["details"],baseline_prob=baseline_prob,modified_cust=res["cust"],
                    top_shap_feat=top_shap_feat,sandbox_sims=n_sims)
                st.rerun()
        else: st.caption("No changes — adjust features above.")
    with rc3: st.caption(f"Simulations: **{n_sims}**")

# ── NO ACTION ──────────────────────────────────────────────────────────────────
st.divider()
btn_lbl="⚠️ Close Case — No Action (High Risk Customer)" if tier=="High" else "✅ No Intervention Needed — Close Case"
if st.button(btn_lbl,key=f"no_action_{display_id}",use_container_width=True,type="secondary"):
    handle_action(action_type="No Action",action_source="None",details="No intervention selected",
        baseline_prob=baseline_prob,modified_cust=None,top_shap_feat=top_shap_feat,
        sandbox_sims=st.session_state.sandbox_sim_count)
    st.rerun()
