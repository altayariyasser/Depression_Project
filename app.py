

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
    --bg0:#080c14;
    --bg1:#111827;
    --panel:#0b101a;
    --card:#151c2e;
    --card2:#1b2338;
    --cream:#fff3d7;
    --peach:#f6c27a;
    --orange:#ee7b42;
    --red:#8b0000;
    --text:#f8fafc;
    --muted:#c8b99d;
}

/* Main app background */
.stApp {
    background:
        radial-gradient(circle at 18% 8%, rgba(238,123,66,0.10), transparent 28%),
        linear-gradient(170deg,#080c14 0%,#111827 100%);
    color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0f19 0%,#141a29 100%);
    border-right: 1px solid rgba(246,194,122,0.12);
}
section[data-testid="stSidebar"] * { color:#f8fafc; }

/* Headings and text */
h1,h2,h3,h4 { color:#f8fafc; letter-spacing:-0.02em; }
p, li, label, span { color:#f4f1e8; }

/* Cards */
.metric-card, .prep-card {
    background: linear-gradient(135deg,#151c2e 0%,#1d263d 100%);
    border-radius:18px;
    padding:22px 24px;
    border:1px solid rgba(246,194,122,0.22);
    box-shadow:0 8px 28px rgba(0,0,0,0.38);
}
.metric-card h3, .prep-card .card-main { color:#fff3d7 !important; }
.metric-card span, .prep-card .card-title, .prep-card .card-detail { color:#c8b99d !important; }
.prep-card .card-sub { color:#f6c27a !important; }

.section-card {
    background: linear-gradient(135deg,rgba(21,28,46,0.94),rgba(11,16,26,0.92));
    border-radius:18px;
    padding:22px 26px;
    border:1px solid rgba(246,194,122,0.16);
    box-shadow:0 8px 24px rgba(0,0,0,0.32);
    margin-bottom:18px;
    color:#f4f1e8;
}
.section-card b { color:#fff3d7; }

.badge {
    display:inline-block;
    padding:5px 13px;
    border-radius:999px;
    background:rgba(238,123,66,0.16);
    color:#f6c27a;
    font-size:.78rem;
    font-weight:700;
    letter-spacing:.03em;
    margin-right:6px;
}

/* Tabs */
button[data-baseweb="tab"] {
    background:rgba(255,243,215,0.04);
    border-radius:12px 12px 0 0;
    margin-right:4px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background:rgba(238,123,66,0.16);
    border-bottom:2px solid #ee7b42;
}

/* Inputs */
.stSelectbox, .stSlider, .stNumberInput, .stRadio {
    color:#f8fafc;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border:1px solid rgba(246,194,122,0.13);
    border-radius:14px;
    overflow:hidden;
}

/* Buttons */
.stButton>button {
    background:linear-gradient(135deg,#ee7b42,#8b0000);
    color:white;
    border:none;
    border-radius:14px;
    font-weight:700;
    box-shadow:0 6px 22px rgba(139,0,0,0.28);
}
.stButton>button:hover {
    border:none;
    transform:translateY(-1px);
}

/* Info/success/error boxes soften */
.stAlert {
    background:rgba(21,28,46,0.88);
    border:1px solid rgba(246,194,122,0.16);
    border-radius:16px;
}
</style>
""", unsafe_allow_html=True)

# ─── THEME COLORS ────────────────────────────────────────────────────────────
# Warm dark dashboard palette inspired by the Stress Level chart
C_BG      = "#080c14"
C_PANEL   = "#0b101a"
C_CARD    = "#151c2e"
C_CREAM   = "#fff3d7"
C_PEACH   = "#f6c27a"
C_ORANGE  = "#ee7b42"
C_RED     = "#8b0000"
C_DARKRED = "#5f0000"

C_NO   = C_CREAM      # No Depression
C_YES  = C_RED        # Depression
C_GRAD = [C_CREAM, C_PEACH, C_ORANGE, C_RED]
WARM_SCALE = [
    [0.00, C_CREAM],
    [0.25, C_PEACH],
    [0.55, C_ORANGE],
    [0.78, "#c72323"],
    [1.00, C_RED],
]
WARM_DIVERGING = [
    [0.00, C_CREAM],
    [0.50, C_PEACH],
    [1.00, C_RED],
]

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C_PANEL,
    font=dict(color="#f8fafc"),
    title_font=dict(color="#fff3d7"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.13)", zerolinecolor="rgba(255,255,255,0.20)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.13)", zerolinecolor="rgba(255,255,255,0.20)"),
)

NUMERIC_FEATURES     = ["Age","CGPA","Sleep_Duration","Study_Hours",
                         "Social_Media_Hours","Physical_Activity","Stress_Level"]
CATEGORICAL_FEATURES = ["Gender","Department"]
DEPARTMENTS = ["Computer Science","Business","Engineering","Medicine","Arts","Psychology"]
GENDERS     = ["Male","Female"]

# ─── DATA ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dataset(n=100_000, sample_n=20_000, seed=42):
    rng = np.random.default_rng(seed)
    age   = rng.integers(18, 25, n)
    gender = rng.choice(GENDERS, n)
    dept   = rng.choice(DEPARTMENTS, n, p=[.22,.20,.20,.16,.12,.10])
    cgpa   = np.clip(rng.normal(3.0, 0.5, n), 0, 4).round(2)
    sleep  = np.clip(rng.normal(6.5, 1.4, n), 2, 11).round(1)
    study  = np.clip(rng.normal(4.5, 1.8, n), 0, 12).round(1)
    social = np.clip(rng.normal(3.2, 1.6, n), 0, 9).round(1)
    phys   = np.clip(rng.normal(74, 28, n), 0, 150).round(0)
    stress = np.clip(rng.normal(5 - .3*(sleep-6.5), 2.0, n), 1, 10).round(0).astype(int)

    z = (-1.1*(cgpa-3.0) + .32*(stress-5) - .18*(sleep-6.5)
         + .12*(social-3.2) - .01*(phys-74) + rng.normal(0, 1.0, n) - 2.8)
    dep = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)

    full = pd.DataFrame({
        "Student_ID":        [f"S{100000+i}" for i in range(n)],
        "Age":               age, "Gender":gender, "Department":dept,
        "CGPA":              cgpa, "Sleep_Duration":sleep,
        "Study_Hours":       study, "Social_Media_Hours":social,
        "Physical_Activity": phys, "Stress_Level":stress,
        "Depression":        dep,
    })

    # Stratified sample: preserve Depression class ratio exactly
    from sklearn.model_selection import train_test_split
    sampled, _ = train_test_split(
        full, train_size=sample_n, stratify=full["Depression"], random_state=seed
    )
    return sampled.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_all_models(df):
    dc = df.drop(columns=["Student_ID"])
    X  = dc.drop(columns=["Depression"])
    y  = dc["Depression"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=30, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(),                         NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first",
                              handle_unknown="ignore"),   CATEGORICAL_FEATURES),
    ])
    X_tr = pre.fit_transform(X_train)
    X_te = pre.transform(X_test)

    pca       = PCA(n_components=10, random_state=42)
    X_tr_pca  = pca.fit_transform(X_tr)
    X_te_pca  = pca.transform(X_te)

    configs = {
        "Logistic Regression":          (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), X_tr,     X_te),
        "Random Forest":                (RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42), X_tr,     X_te),
        "Logistic Regression + PCA":    (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), X_tr_pca, X_te_pca),
        "Random Forest + PCA":          (RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42), X_tr_pca, X_te_pca),
    }

    results = {}
    for name, (mdl, Xtr, Xte) in configs.items():
        mdl.fit(Xtr, y_train)
        pred  = mdl.predict(Xte)
        proba = mdl.predict_proba(Xte)[:,1]
        results[name] = dict(
            model=mdl, pred=pred, proba=proba,
            accuracy =accuracy_score(y_test, pred),
            precision=precision_score(y_test, pred),
            recall   =recall_score(y_test, pred),
            f1       =f1_score(y_test, pred),
            cm       =confusion_matrix(y_test, pred),
            report   =classification_report(y_test, pred, output_dict=True),
        )

    return dict(
        preprocessor=pre, pca=pca, results=results,
        X_test=X_test, y_test=y_test,
        X_te_processed=X_te, X_te_pca=X_te_pca,
    )


with st.spinner("Loading data & training models…"):
    df     = generate_dataset()
    bundle = train_all_models(df)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("##  Navigator")
page = st.sidebar.radio("", [
    "📊 Data Overview",
    "🔍 EDA",
    "🧹 Cleaning & Preprocessing",
    "🤖 Model Results",
    "🎯 Interactive Prediction",
], label_visibility="collapsed")
st.sidebar.markdown("---")

st.title(" Student Lifestyle & Depression Prediction")
st.caption("An interactive walkthrough of the full ML pipeline — from raw data to a live prediction.")

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Overview":
    st.markdown("## 📊 Data Overview")
    st.markdown(
        "<div class='section-card'>This project analyses <b>student lifestyle factors</b> "
        "(sleep, study habits, social-media use, physical activity, stress) to predict "
        "whether a student is likely to experience <b>depression</b>. "
        "The target is binary and notably <b>imbalanced</b>.</div>",
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1,c2,c3,c4],
        ["Rows","Columns","Depression Rate","Departments"],
        [f"{df.shape[0]:,}", str(df.shape[1]),
         f"{df['Depression'].mean()*100:.1f}%", str(df['Department'].nunique())]
    ):
        col.markdown(
            f"<div class='metric-card'><h3 style='margin:0'>{val}</h3>"
            f"<span style='color:#9aa3c0'>{label}</span></div>",
            unsafe_allow_html=True)

    st.markdown("### Sample of the Dataset")
    st.dataframe(df.sample(10, random_state=1).reset_index(drop=True), use_container_width=True)

    st.markdown("### Column Descriptions")
    st.dataframe(pd.DataFrame({
        "Column":["Student_ID","Age","Gender","Department","CGPA","Sleep_Duration",
                  "Study_Hours","Social_Media_Hours","Physical_Activity","Stress_Level","Depression"],
        "Type":  ["ID","int","cat","cat","float","float","float","float","float","int","int (target)"],
        "Description":[
            "Unique identifier — dropped before modelling",
            "Student age (18–24)",
            "Male / Female",
            "Academic department",
            "Cumulative GPA (0–4 scale)",
            "Average nightly sleep in hours",
            "Average daily study hours",
            "Average daily social-media hours",
            "Weekly physical activity (minutes)",
            "Self-reported stress level (1–10)",
            "1 = depressed · 0 = not depressed",
        ],
    }), use_container_width=True, hide_index=True)

    # ── Target Distribution & Feature Impact ────────────────────────────────
    st.markdown("### Target Distribution & Feature Impact on Depression")
    c1, c2 = st.columns([1.4, 1])

    with c1:
        from scipy.stats import gaussian_kde

        dep0 = df.loc[df["Depression"]==0, "CGPA"].values
        dep1 = df.loc[df["Depression"]==1, "CGPA"].values

        xs = np.linspace(0, 4, 300)
        kde0 = gaussian_kde(dep0, bw_method=0.25)(xs)
        kde1 = gaussian_kde(dep1, bw_method=0.25)(xs)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=kde0, mode="lines",
            name="No Depression",
            line=dict(color=C_NO, width=2.5),
            fill="tozeroy", fillcolor="rgba(255,243,215,0.30)",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=kde1, mode="lines",
            name="Depression",
            line=dict(color=C_RED, width=2.5),
            fill="tozeroy", fillcolor="rgba(139,0,0,0.30)",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
            title="Distribution of CGPA by Depression Status",
            xaxis_title="CGPA", yaxis_title="Density",
            legend=dict(orientation="h", y=1.12, x=0),
            height=330,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ov = df.copy()
        for col in CATEGORICAL_FEATURES:
            ov[col] = ov[col].astype("category").cat.codes
        impact = (ov[NUMERIC_FEATURES+CATEGORICAL_FEATURES]
                  .corrwith(ov["Depression"].astype(int))
                  .reset_index())
        impact.columns = ["Feature","Effect"]
        impact = impact.sort_values("Effect")
        fig = px.bar(
            impact, x="Effect", y="Feature", orientation="h",
            color="Effect", color_continuous_scale=WARM_DIVERGING,
            title="Feature → Depression Correlation",
            labels={"Effect":"Pearson r with Depression"},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL, height=310)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown("## 🔍 Exploratory Data Analysis")
    t1, t2, t3, t4 = st.tabs(
        ["Target & Distributions","Stress & Sleep","CGPA & Department","Correlations"])

    # ── Tab 1 ──
    with t1:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            counts = df["Depression"].value_counts().rename({0:"No Depression",1:"Depression"})
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type":"pie"},{"type":"bar"}]],
                subplot_titles=["Class Split","Count"]
            )
            fig.add_trace(go.Pie(
                values=counts.values, labels=counts.index, hole=0.55,
                marker_colors=[C_NO, C_YES],
                textinfo="percent", pull=[0,.08],
                showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[C_NO, C_YES],
                text=[f"{v:,}" for v in counts.values],
                textposition="outside", showlegend=False,
            ), row=1, col=2)
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
                title_text="Depression Distribution", height=340,
                legend=dict(orientation="h", y=-0.12),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            feat = st.selectbox("Choose a feature to explore by Depression status", NUMERIC_FEATURES, index=1)
            dep_label = df["Depression"].map({0:"No Depression",1:"Depression"})
            fig = px.histogram(
                df, x=feat, color=dep_label, barmode="overlay",
                nbins=35, opacity=0.82,
                color_discrete_map={"No Depression": C_NO, "Depression": C_YES},
                title=f"{feat} — Distribution by Depression Status",
                labels={feat: feat, "color":""},
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
                              legend_title="", height=340)
            st.plotly_chart(fig, use_container_width=True)

        dep_pct = df["Depression"].mean() * 100
        no_dep_pct = 100 - dep_pct
        st.markdown(f"""
        <div class='section-card'>
        <b>💡 Depression Distribution Insight</b><br><br>
        The target variable is <b>imbalanced</b>. Most students are in the
        <b>No Depression</b> class (<b>{no_dep_pct:.1f}%</b>), while the Depression class is smaller
        (<b>{dep_pct:.1f}%</b>). This is important because <b>accuracy alone is not enough</b>
        to evaluate the model. A model can look accurate if it mostly predicts the majority class.
        For this reason, we should also focus on <b>precision, recall, and F1-score</b>, especially
        for the Depression class.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2 ──
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            sd = (df.groupby("Stress_Level")["Depression"].mean()*100).reset_index()
            fig = px.bar(sd, x="Stress_Level", y="Depression",
                         color="Depression", color_continuous_scale=WARM_SCALE,
                         title="Depression Rate (%) by Stress Level",
                         labels={"Depression":"Depression Rate (%)"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sg = pd.cut(df["Sleep_Duration"],[0,5,7,9,12],
                        labels=["Low (<5h)","Normal (5-7h)","Good (7-9h)","High (>9h)"])
            sl = (df.groupby(sg)["Depression"].mean()*100).reset_index()
            fig = px.bar(sl, x="Sleep_Duration", y="Depression",
                         color="Depression", color_continuous_scale=WARM_SCALE,
                         title="Depression Rate (%) by Sleep Group",
                         labels={"Depression":"Depression Rate (%)","Sleep_Duration":"Sleep Group"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL)
            st.plotly_chart(fig, use_container_width=True)

        mg = pd.cut(df["Social_Media_Hours"],[0,2,4,6,10],
                    labels=["Low (<2h)","Moderate (2-4h)","High (4-6h)","Very High (>6h)"],
                    include_lowest=True)
        ml = (df.groupby(mg)["Depression"].mean()*100).reset_index()
        fig = px.bar(ml, x="Social_Media_Hours", y="Depression",
                     color="Depression", color_continuous_scale=WARM_SCALE,
                     title="Depression Rate (%) by Social Media Usage",
                     labels={"Depression":"Depression Rate (%)","Social_Media_Hours":"Social Media Group"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL)
        st.plotly_chart(fig, use_container_width=True)

        high_stress = (df.groupby("Stress_Level")["Depression"].mean()*100).iloc[-3:].mean()
        low_stress  = (df.groupby("Stress_Level")["Depression"].mean()*100).iloc[:3].mean()
        st.markdown(f"""
        <div class='section-card'>
        <b>💡 Stress, Sleep, and Social Media Insight</b><br><br>
        Students with higher stress levels tend to show a higher depression rate. In this dataset,
        the high-stress group has an average depression rate of about <b>{high_stress:.1f}%</b>,
        compared with about <b>{low_stress:.1f}%</b> for the lower-stress group. This makes
        <b>Stress_Level</b> an important feature for prediction, and it also makes sense logically
        because higher stress can be connected to worse mental health.
        <br><br>
        Sleep duration and social media hours were grouped only to make the EDA easier to understand.
        These grouped columns help us visualize patterns, but the model can still learn from the
        original continuous features.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3 ──
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            dep_label = df["Depression"].map({0:"No Depression",1:"Depression"})
            fig = px.violin(df, x=dep_label, y="CGPA", color=dep_label,
                            box=True, points=False,
                            color_discrete_map={"No Depression":C_NO,"Depression":C_YES},
                            title="CGPA by Depression Status")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
                              showlegend=False, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            dd = (df.groupby("Department")["Depression"].mean()*100).reset_index()
            fig = px.bar(dd.sort_values("Depression"), x="Depression", y="Department",
                         orientation="h", color="Depression", color_continuous_scale=WARM_SCALE,
                         title="Depression Rate (%) by Department",
                         labels={"Depression":"Depression Rate (%)"})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL)
            st.plotly_chart(fig, use_container_width=True)

        dep_median    = df.loc[df["Depression"]==1, "CGPA"].median()
        nodep_median  = df.loc[df["Depression"]==0, "CGPA"].median()
        dept_spread   = dd["Depression"].max() - dd["Depression"].min()
        st.markdown(f"""
        <div class='section-card'>
        <b>💡 CGPA and Department Insight</b><br><br>
        The CGPA plot shows a clear difference between the two groups. Students with depression have
        a lower median CGPA (<b>{dep_median:.2f}</b>) compared with students without depression
        (<b>{nodep_median:.2f}</b>). This does not prove that CGPA causes depression, but together
        with other lifestyle factors it can give useful information for prediction.
        <br><br>
        Department shows a much weaker relationship with depression. The difference between the highest
        and lowest department depression rates is only about <b>{dept_spread:.1f}</b> percentage points,
        so Department may be less important than features such as CGPA and Stress_Level.
        </div>
        """, unsafe_allow_html=True)
    with t4:
        cd = df.copy()
        cd["Depression"] = cd["Depression"].astype(int)

        corr_with_dep = (
            cd[NUMERIC_FEATURES + ["Depression"]]
            .corr()["Depression"]
            .drop("Depression")
            .sort_values()
            .reset_index()
        )

        corr_with_dep.columns = ["Feature", "Correlation"]

        fig = px.bar(
            corr_with_dep,
            x="Correlation",
            y="Feature",
            orientation="h",
            text="Correlation",
            color="Correlation",
            color_continuous_scale=[
                [0.00, "#f7e7c1"],
                [0.50, "#f6ad55"],
                [1.00, "#8b0000"]
            ],
            title="Feature Correlation with Depression"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f131c",
            height=480,
            xaxis_title="Correlation with Depression",
            yaxis_title="Feature",
            coloraxis_showscale=False,
            font=dict(color="white"),
            xaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="#f6ad55",
                gridcolor="rgba(255,255,255,0.12)"
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)"
            ),
            margin=dict(l=30, r=60, t=70, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        cgpa_r = corr_with_dep.loc[corr_with_dep["Feature"] == "CGPA", "Correlation"].values[0]
        stress_r = corr_with_dep.loc[corr_with_dep["Feature"] == "Stress_Level", "Correlation"].values[0]
        sleep_r = corr_with_dep.loc[corr_with_dep["Feature"] == "Sleep_Duration", "Correlation"].values[0]

        st.markdown(f"""
        <div class='section-card'>
        <b>💡 Correlation Insight</b><br><br>
        This chart focuses only on the relationship between each numerical feature and the target variable,
        <b>Depression</b>.the main goal
        of the project is to understand which features are related to depression.
        <br><br>
        <b>Stress_Level</b> has the strongest positive relationship with Depression
        (<b>r = {stress_r:.2f}</b>), meaning higher stress is associated with a higher depression rate.
        <b>CGPA</b> has a negative relationship with Depression
        (<b>r = {cgpa_r:.2f}</b>), meaning students with higher CGPA tend to have lower depression rates.
        <b>Sleep_Duration</b> also has a negative relationship
        (<b>r = {sleep_r:.2f}</b>), suggesting that students who sleep more tend to have lower depression rates.
        <br><br>
        Overall, the correlations are not very strong. This means depression is likely influenced by a
        combination of features rather than one single variable.
        </div>
        """, unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# 3 — CLEANING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧹 Cleaning & Preprocessing":
    st.markdown("## 🧹 Cleaning & Preprocessing")

    # ── 4 descriptive cards ──────────────────────────────────────────────────
    card_css = """
    <style>
    .prep-card {
        background: linear-gradient(135deg,#151c2e 0%,#1d263d 100%);
        border-radius: 18px; padding: 20px 22px;
        border: 1px solid rgba(246,194,122,0.22);
        box-shadow: 0 8px 26px rgba(0,0,0,0.35);
        height: 100%;
    }
    .prep-card .card-title  { color:#c8b99d; font-size:.82rem; font-weight:700;
                               letter-spacing:.05em; text-transform:uppercase; margin-bottom:8px; }
    .prep-card .card-main   { color:#fff3d7; font-size:1.35rem; font-weight:800; margin-bottom:6px; }
    .prep-card .card-sub    { color:#f6c27a; font-size:.82rem; }
    .prep-card .card-detail { color:#c8b99d; font-size:.78rem; margin-top:4px; }
    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Train / Test Split",     "80 / 20",           "Stratified",          "random_state = 30"),
        ("Categorical Encoding",   "One-Hot Encoder",   "Gender, Department",  "drop = 'first'"),
        ("Numerical Scaling",      "StandardScaler",    "7 numeric features",  "fit on train only"),
        ("PCA Output",             "10 Components",     "≈ 95% variance",      "applied post-scaling"),
    ]
    for col, (title, main, sub, detail) in zip([c1,c2,c3,c4], cards):
        col.markdown(
            f"<div class='prep-card'>"
            f"<div class='card-title'>{title}</div>"
            f"<div class='card-main'>{main}</div>"
            f"<div class='card-sub'>{sub}</div>"
            f"<div class='card-detail'>{detail}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Before vs After — Scaling")
    sample_raw = df[NUMERIC_FEATURES].sample(5, random_state=7).reset_index(drop=True)
    sc_prev = StandardScaler().fit(df[NUMERIC_FEATURES])
    sample_sc = pd.DataFrame(sc_prev.transform(sample_raw), columns=NUMERIC_FEATURES).round(3)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw values**"); st.dataframe(sample_raw, use_container_width=True)
    with c2:
        st.markdown("**After StandardScaler**"); st.dataframe(sample_sc, use_container_width=True)

    st.markdown("### Before vs After — One-Hot Encoding")
    sample_cat = df[CATEGORICAL_FEATURES].sample(5, random_state=7).reset_index(drop=True)
    ohe_prev = OneHotEncoder(drop="first", handle_unknown="ignore").fit(df[CATEGORICAL_FEATURES])
    enc_cols  = ohe_prev.get_feature_names_out(CATEGORICAL_FEATURES)
    sample_enc = pd.DataFrame(ohe_prev.transform(sample_cat).toarray().astype(int), columns=enc_cols)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw categories**"); st.dataframe(sample_cat, use_container_width=True)
    with c2:
        st.markdown("**After OneHotEncoding**"); st.dataframe(sample_enc, use_container_width=True)

    # PCA
    st.markdown("### Principal Component Analysis (PCA)")
    X_all = df.drop(columns=["Student_ID","Depression"])
    X_proc = bundle["preprocessor"].transform(X_all)
    pca_full = PCA().fit(X_proc)
    ev  = pca_full.explained_variance_ratio_
    cev = np.cumsum(ev)
    n95 = int(np.argmax(cev >= .95) + 1)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(ev)+1)), y=ev*100,
            marker_color=[C_ORANGE if i < n95 else "rgba(255,255,255,0.12)"
                          for i in range(len(ev))],
            name="Individual",
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cev)+1)), y=cev*100,
            mode="lines+markers", line=dict(color=C_RED, width=2.5),
            marker=dict(size=6), name="Cumulative",
            yaxis="y2",
        ))
        fig.add_hline(y=95, line_dash="dash", line_color="#f6ad55",
                      annotation_text="95% threshold", yref="y2",
                      annotation_position="bottom right")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
            title=f"Explained Variance per Component — {n95} components reach 95%",
            xaxis_title="Principal Component",
            yaxis=dict(title="Individual variance (%)"),
            yaxis2=dict(title="Cumulative variance (%)", overlaying="y",
                        side="right", range=[0, 105]),
            legend=dict(orientation="h", y=1.12), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pca2 = PCA(n_components=2).fit_transform(X_proc)
        sdf  = pd.DataFrame({"PC1":pca2[:,0], "PC2":pca2[:,1],
                              "Depression": df["Depression"].map({0:"No Depression",1:"Depression"})})
        fig = px.scatter(
            sdf.sample(min(5000, len(sdf)), random_state=1),
            x="PC1", y="PC2", color="Depression", opacity=0.55,
            color_discrete_map={"No Depression":C_NO,"Depression":C_YES},
            title="Students in 2-D PCA Space",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
                          legend_title="", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"The two classes overlap heavily in 2-D PCA space — depression isn't linearly separable "
        f"along the top 2 components, which is why the full {n95}-component version and non-linear "
        "models."
    )
    st.markdown("""
<div class='section-card'>
<b>💡 Cleaning & Preprocessing Summary</b><br><br>

During the cleaning and preprocessing step, unnecessary columns were removed before modeling. 
The <b>Student_ID</b> column was dropped because it is only an identifier and does not provide useful information for predicting depression.

The columns <b>Social_Media_Group</b> and <b>Sleep_Duration_Group</b> were also removed because they were created only for EDA visualization. 
They helped us understand patterns in the data, but they were not used as original modeling features.

Categorical features such as <b>Gender</b> and <b>Department</b> were transformed using <b>One-Hot Encoding</b>. 
This was necessary because machine learning models cannot directly understand text categories. 
One-Hot Encoding also avoids creating a false order between categories.

The target variable <b>Depression</b> was prepared as a numeric classification target, where <b>1</b> represents depression and <b>0</b> represents no depression.

<b>StandardScaler</b> was applied to the numerical features because they have different scales. 
For example, <b>Physical_Activity</b> ranges from 0 to 150, while features such as <b>CGPA</b>, <b>Sleep_Duration</b>, and <b>Study_Hours</b> have much smaller ranges. 
Scaling helps prevent large-scale features from dominating the model.

The data was split into training and testing sets using a stratified split. 
Stratification was used to make sure both sets had a similar proportion of depression and non-depression cases.

The split was done before fitting the scaler and encoder to avoid data leakage. 
This means the scaler and encoder learned only from the training data, and then the same transformations were applied to the test data.
</div>
""", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# 4 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown("## 🤖 Model Results")
    results = bundle["results"]

    st.markdown(
        "<div class='section-card'>Four models were trained: Logistic Regression and Random Forest, "
        "each with and without PCA (10 components). All use <code>class_weight='balanced'</code> to handle "
        "the imbalanced target. <b>Recall</b> and <b>F1-score</b> on class 1 are the key metrics.</div>",
        unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        name: {"Accuracy":r["accuracy"],"Precision":r["precision"],
               "Recall":r["recall"],"F1-Score":r["f1"]}
        for name, r in results.items()
    }).T.reset_index().rename(columns={"index":"Model"})

    melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_sequence=C_GRAD, text_auto=".2f",
                 title="All 4 Models — Performance Comparison")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL, yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    model_choice = st.selectbox("Inspect a model in detail", list(results.keys()))
    r = results[model_choice]

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1,c2,c3,c4],
        ["Accuracy","Precision","Recall","F1-Score"],
        [r["accuracy"],r["precision"],r["recall"],r["f1"]]):
        col.markdown(
            f"<div class='metric-card'><h3 style='margin:0'>{val:.2%}</h3>"
            f"<span style='color:#9aa3c0'>{label}</span></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        cm = r["cm"]
        fig = px.imshow(
            cm, text_auto=True,
            color_continuous_scale=WARM_SCALE,
            x=["Pred: No Depression","Pred: Depression"],
            y=["Actual: No Depression","Actual: Depression"],
            title=f"Confusion Matrix — {model_choice}",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rpt = pd.DataFrame(r["report"]).T.round(3)
        rpt = rpt.rename(index={"0":"No Depression","1":"Depression"})
        st.markdown("**Classification Report**")
        st.dataframe(rpt, use_container_width=True)

        if "PCA" not in model_choice and "Random Forest" in model_choice:
            imp = r["model"].feature_importances_
            fn  = [s.replace("num__","").replace("cat__","")
                   for s in bundle["preprocessor"].get_feature_names_out()]
            idf = pd.DataFrame({"Feature":fn,"Importance":imp}).sort_values("Importance").tail(10)
            fig = px.bar(idf, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=WARM_SCALE,
                         title="Top 10 Feature Importances")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL)
            st.plotly_chart(fig, use_container_width=True)
        elif "PCA" not in model_choice and "Logistic" in model_choice:
            coefs = r["model"].coef_[0]
            fn    = [s.replace("num__","").replace("cat__","")
                     for s in bundle["preprocessor"].get_feature_names_out()]
            cdf = pd.DataFrame({"Feature":fn,"Coefficient":coefs}).sort_values("Coefficient")
            fig = px.bar(cdf, x="Coefficient", y="Feature", orientation="h",
                         color="Coefficient",
                         color_continuous_scale=WARM_DIVERGING,
                         title="Logistic Regression Coefficients")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature-level importance is shown for the non-PCA models (PCA transforms features into components).")
    st.markdown("""
<div class='section-card'>
<h2>Best Model</h2>

<p>
The best model was selected based mainly on <b>F1-score</b> and <b>recall</b>, not accuracy alone.
This is because the project focuses on predicting depression, and detecting actual depression cases is more important than only achieving high accuracy.
</p>

<p>
Logistic Regression gave better results for <b>recall</b>. This is useful because recall measures how many actual depression cases the model correctly detected.
Since the dataset is imbalanced and the depression class has fewer samples, recall is an important metric for this project.
</p>

<p>
Random Forest achieved higher accuracy, but its recall was lower. This means that although it performed well overall, it missed more students who actually had depression.
This shows why accuracy alone can be misleading in an imbalanced classification problem.
</p>

<p>
PCA was applied using 10 components for both models. However, the results with PCA were very similar to the results without PCA.
This means PCA did not significantly improve model performance in this project.
</p>

<p>
Overall, <b>Logistic Regression</b> is preferred when the goal is to detect more depression cases, while Random Forest may look better if we only focus on accuracy.
</p>
</div>
""", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# 5 — INTERACTIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Interactive Prediction":
    st.markdown("## 🎯 Interactive Prediction")
    st.markdown(
        "<div class='section-card'>Fill in a student's lifestyle profile below and get a live "
        "risk prediction from your chosen trained model.</div>", unsafe_allow_html=True)

    model_name = st.radio("Choose model", list(bundle["results"].keys()), horizontal=True)
    use_pca    = "PCA" in model_name
    model      = bundle["results"][model_name]["model"]
    pre        = bundle["preprocessor"]
    pca        = bundle["pca"]

    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.slider("Age", 18, 24, 20)
        gender = st.selectbox("Gender", GENDERS)
        dept   = st.selectbox("Department", DEPARTMENTS)
    with c2:
        cgpa   = st.slider("CGPA", 0.0, 4.0, 3.0, 0.05)
        sleep  = st.slider("Sleep Duration (hours/night)", 2.0, 11.0, 6.5, 0.5)
        study  = st.slider("Study Hours (per day)", 0.0, 12.0, 4.5, 0.5)
    with c3:
        social = st.slider("Social Media Hours (per day)", 0.0, 9.0, 3.0, 0.5)
        phys   = st.number_input("Physical Activity (min/week)", 0, 150, 74)
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)

    if st.button("🔮 Predict Depression Risk", use_container_width=True):
        inp = pd.DataFrame([{
            "Age":age,"Gender":gender,"Department":dept,"CGPA":cgpa,
            "Sleep_Duration":sleep,"Study_Hours":study,
            "Social_Media_Hours":social,"Physical_Activity":phys,"Stress_Level":stress,
        }])
        X_p = pre.transform(inp)
        if use_pca:
            X_p = pca.transform(X_p)

        pred  = model.predict(X_p)[0]
        proba = model.predict_proba(X_p)[0][1]

        c1, c2 = st.columns([1, 1.3])
        with c1:
            gauge_color = C_YES if pred == 1 else C_NO
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                number={"suffix":"%","font":{"size":38}},
                title={"text":"Depression Risk","font":{"size":18}},
                gauge={
                    "axis":{"range":[0,100],"tickwidth":1},
                    "bar":{"color":gauge_color, "thickness":0.25},
                    "steps":[
                        {"range":[0,33],  "color":"#1a3a2a"},
                        {"range":[33,66], "color":"#3a3010"},
                        {"range":[66,100],"color":"#3a1020"},
                    ],
                    "threshold":{"line":{"color":"white","width":3},"value":50},
                },
            ))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL, height=320)
            st.plotly_chart(fig, use_container_width=True)

            if pred == 1:
                st.error(f"⚠️ **Depression predicted** — risk score {proba:.1%}")
            else:
                st.success(f"✅ **No depression predicted** — risk score {proba:.1%}")

        with c2:
            avg = df[NUMERIC_FEATURES].mean()
            feats = ["CGPA","Sleep_Duration","Study_Hours","Social_Media_Hours","Stress_Level"]
            student_vals = [cgpa, sleep, study, social, stress]
            avg_vals     = [avg[f] for f in feats]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=student_vals, theta=feats, fill="toself",
                name="This Student", line_color=C_RED, fillcolor="rgba(139,0,0,0.22)"))
            fig.add_trace(go.Scatterpolar(
                r=avg_vals, theta=feats, fill="toself",
                name="Dataset Average", line_color=C_CREAM, fillcolor="rgba(255,243,215,0.18)"))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_PANEL,
                title="Student vs Dataset Average", height=340,
                legend=dict(orientation="h", y=-0.12),
            )
            st.plotly_chart(fig, use_container_width=True)
