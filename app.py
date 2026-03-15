import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import io

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.stApp {
    background: #0d0d14;
    color: #e8e4f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13131f !important;
    border-right: 1px solid #2a2a3d;
}

/* Main header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c77dff 0%, #7b2fff 50%, #48cae4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: #8b8b9e;
    font-size: 1rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 0.8rem;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #7b7b94;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #c77dff;
}
.metric-card .unit {
    font-size: 0.75rem;
    color: #8b8b9e;
}

/* Section headers */
.section-tag {
    display: inline-block;
    background: rgba(199,125,255,0.12);
    color: #c77dff;
    border: 1px solid rgba(199,125,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.6rem;
}

/* Persona badges */
.persona-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2a4a, #c77dff44, #2a2a4a, transparent);
    margin: 1.5rem 0;
}

/* Cluster table */
.cluster-table {
    background: #13131f;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #2a2a3d;
}

/* Upload area */
[data-testid="stFileUploadDropzone"] {
    background: #13131f !important;
    border: 2px dashed #2a2a4a !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7b2fff, #c77dff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    transition: all 0.2s;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(123,47,255,0.4);
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: #13131f !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #8b8b9e !important;
    border-radius: 8px !important;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, #7b2fff22, #c77dff22) !important;
    color: #c77dff !important;
}

/* Selectbox / sliders */
[data-baseweb="select"] > div {
    background: #13131f !important;
    border-color: #2a2a4a !important;
    color: #e8e4f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
PERSONA_MAP = {
    "High income, High spending":  ("Luxury Shoppers",      "#c77dff", "#2a1040"),
    "High income, Low spending":   ("Affluent Minimalists",  "#48cae4", "#0d2730"),
    "Low income, High spending":   ("Aspirational Spenders", "#f77f00", "#2e1800"),
    "Low income, Low spending":    ("Frugal Segment",        "#8b8b9e", "#1a1a2e"),
}

def label_cluster(row, income_mean, spend_mean):
    if row["Annual Income (k$)"] >= income_mean and row["Spending Score (1-100)"] >= spend_mean:
        return "High income, High spending"
    if row["Annual Income (k$)"] >= income_mean and row["Spending Score (1-100)"] < spend_mean:
        return "High income, Low spending"
    if row["Annual Income (k$)"] < income_mean and row["Spending Score (1-100)"] >= spend_mean:
        return "Low income, High spending"
    return "Low income, Low spending"

def safe_metrics(X, labels):
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X):
        return np.nan, np.nan, np.nan
    return (silhouette_score(X, labels),
            davies_bouldin_score(X, labels),
            calinski_harabasz_score(X, labels))

def dark_style():
    """Return common matplotlib dark style kwargs."""
    return {
        "facecolor": "#0d0d14",
        "text.color": "#e8e4f0",
        "axes.facecolor": "#13131f",
        "axes.edgecolor": "#2a2a4a",
        "xtick.color": "#8b8b9e",
        "ytick.color": "#8b8b9e",
        "axes.labelcolor": "#e8e4f0",
        "grid.color": "#1e1e30",
        "axes.titlecolor": "#e8e4f0",
    }

def apply_dark(fig, axes_list=None):
    s = dark_style()
    fig.patch.set_facecolor(s["facecolor"])
    for ax in (axes_list or fig.axes):
        ax.set_facecolor(s["axes.facecolor"])
        ax.tick_params(colors=s["xtick.color"])
        ax.xaxis.label.set_color(s["axes.labelcolor"])
        ax.yaxis.label.set_color(s["axes.labelcolor"])
        ax.title.set_color(s["axes.titlecolor"])
        for spine in ax.spines.values():
            spine.set_edgecolor(s["axes.edgecolor"])
        ax.grid(color=s["grid.color"], linewidth=0.6)


CLUSTER_CMAP = ["#c77dff", "#48cae4", "#f77f00", "#06d6a0", "#ef233c",
                "#ffd166", "#7209b7", "#3a86ff", "#fb8500"]

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🛍️ Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.8rem">Mall Customer Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-tag">📂 Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Mall_Customers.csv", type="csv", label_visibility="collapsed")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">⚙️ K-Means</div>', unsafe_allow_html=True)

    auto_k = st.toggle("Auto-select best k", value=True)
    manual_k = st.slider("Manual k", 2, 10, 5, disabled=auto_k)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">🔬 DBSCAN</div>', unsafe_allow_html=True)

    run_dbscan = st.toggle("Run DBSCAN", value=True)
    eps_val    = st.slider("eps", 0.1, 2.0, 0.8, 0.1, disabled=not run_dbscan)
    min_samp   = st.slider("min_samples", 2, 20, 10,    disabled=not run_dbscan)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis", use_container_width=True)


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_obj):
    return pd.read_csv(file_obj)

def get_demo_data():
    """Generate synthetic demo data if no file uploaded."""
    np.random.seed(42)
    n = 200
    income  = np.concatenate([np.random.normal(25,6,40), np.random.normal(55,10,80),
                               np.random.normal(87,8,40), np.random.normal(26,5,20), np.random.normal(86,8,20)])
    spend   = np.concatenate([np.random.normal(20,8,40), np.random.normal(50,12,80),
                               np.random.normal(17,8,40), np.random.normal(79,8,20), np.random.normal(82,8,20)])
    income  = np.clip(income, 15, 137).astype(int)
    spend   = np.clip(spend, 1, 99).astype(int)
    gender  = np.random.choice(["Male","Female"], n)
    age     = np.random.randint(18, 70, n)
    return pd.DataFrame({"CustomerID": range(1,n+1), "Genre": gender,
                          "Age": age, "Annual Income (k$)": income,
                          "Spending Score (1-100)": spend})

if uploaded:
    df_raw = load_data(uploaded)
    st.sidebar.success(f"✅ {len(df_raw)} rows loaded")
else:
    df_raw = get_demo_data()
    st.sidebar.info("🎲 Using synthetic demo data")


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">Mall Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Cluster shoppers by income & spending behaviour · K-Means + DBSCAN</div>', unsafe_allow_html=True)
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  VALIDATE COLUMNS
# ─────────────────────────────────────────────
REQUIRED = ["Annual Income (k$)", "Spending Score (1-100)"]
if not all(c in df_raw.columns for c in REQUIRED):
    st.error(f"CSV must contain columns: {REQUIRED}")
    st.stop()

df = df_raw.copy()
X  = df[REQUIRED].copy()

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_overview, tab_kmeans, tab_dbscan, tab_compare, tab_data = st.tabs([
    "📊 Overview", "🎯 K-Means", "🔬 DBSCAN", "⚖️ Comparison", "📋 Data"
])


# ══════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab_overview:
    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Customers",      len(df),                             ""),
        ("Avg Income",     f"{df['Annual Income (k$)'].mean():.1f}", "k$"),
        ("Avg Spending",   f"{df['Spending Score (1-100)'].mean():.1f}", "/ 100"),
        ("Income Range",   f"{df['Annual Income (k$)'].max()-df['Annual Income (k$)'].min()}", "k$"),
    ]
    for col, (lbl, val, unit) in zip([c1,c2,c3,c4], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{lbl}</div>
                <div class="value">{val}</div>
                <div class="unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-tag">Distribution</div>', unsafe_allow_html=True)
        st.markdown("#### Income & Spending Distributions")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        fig.patch.set_facecolor("#0d0d14")
        for ax, col_name, color in zip(axes, REQUIRED, ["#c77dff", "#48cae4"]):
            ax.set_facecolor("#13131f")
            ax.hist(df[col_name], bins=20, color=color, alpha=0.85, edgecolor="#0d0d14", linewidth=0.5)
            ax.axvline(df[col_name].mean(), color="white", lw=1.5, linestyle="--", alpha=0.7, label="mean")
            ax.set_title(col_name.replace(" (k$)", "").replace(" (1-100)",""), color="#e8e4f0", fontsize=10, pad=8)
            ax.tick_params(colors="#8b8b9e", labelsize=8)
            ax.grid(color="#1e1e30", lw=0.5, axis="y")
            for sp in ax.spines.values(): sp.set_color("#2a2a4a")
            ax.legend(fontsize=7, framealpha=0, labelcolor="#8b8b9e")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown('<div class="section-tag">Raw Scatter</div>', unsafe_allow_html=True)
        st.markdown("#### Income vs Spending")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0d0d14")
        ax.set_facecolor("#13131f")
        ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
                   color="#7b2fff", alpha=0.5, edgecolor="#c77dff", linewidth=0.4, s=35)
        ax.set_xlabel("Annual Income (k$)", color="#e8e4f0", fontsize=9)
        ax.set_ylabel("Spending Score", color="#e8e4f0", fontsize=9)
        ax.tick_params(colors="#8b8b9e", labelsize=8)
        ax.grid(color="#1e1e30", lw=0.5)
        for sp in ax.spines.values(): sp.set_color("#2a2a4a")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=np.number).drop(columns=["CustomerID"], errors="ignore")
    if len(numeric_cols.columns) >= 2:
        st.markdown('<div class="section-tag">Correlations</div>', unsafe_allow_html=True)
        st.markdown("#### Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#0d0d14")
        ax.set_facecolor("#13131f")
        corr = numeric_cols.corr()
        cmap = mcolors.LinearSegmentedColormap.from_list("purp", ["#48cae4","#13131f","#c77dff"])
        sns.heatmap(corr, ax=ax, cmap=cmap, annot=True, fmt=".2f",
                    annot_kws={"size": 8, "color": "#e8e4f0"},
                    linecolor="#1a1a2e", linewidths=0.5,
                    cbar_kws={"shrink": 0.7})
        ax.tick_params(colors="#8b8b9e", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════
#  PREPROCESSING  (shared by all tabs)
# ══════════════════════════════════════════════
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca      = PCA(n_components=2, random_state=42)
X_pca    = pca.fit_transform(X_scaled)


# ══════════════════════════════════════════════
#  K-MEANS COMPUTATION
# ══════════════════════════════════════════════
ks, inertias, silhouettes = [], [], []
for k in range(2, 11):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    ks.append(k)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lbl))

best_k = ks[int(np.argmax(silhouettes))] if auto_k else manual_k

kmeans    = KMeans(n_clusters=best_k, random_state=42, n_init=20)
k_labels  = kmeans.fit_predict(X_scaled)
df["KMeans_Cluster"] = k_labels

k_sil = silhouette_score(X_scaled, k_labels)
k_db  = davies_bouldin_score(X_scaled, k_labels)
k_ch  = calinski_harabasz_score(X_scaled, k_labels)

# Cluster summary
inc_mean   = df["Annual Income (k$)"].mean()
spend_mean = df["Spending Score (1-100)"].mean()
summary    = df.groupby("KMeans_Cluster")[REQUIRED].mean().round(2)
counts     = df.groupby("KMeans_Cluster").size().rename("count")
cluster_df = pd.concat([counts, summary], axis=1).reset_index().rename(columns={"KMeans_Cluster":"cluster"})
cluster_df["label"]   = cluster_df.apply(label_cluster, axis=1, args=(inc_mean, spend_mean))
cluster_df["persona"] = cluster_df["label"].map(lambda l: PERSONA_MAP.get(l, ("Unknown","#8b8b9e","#1a1a2e"))[0])


# ══════════════════════════════════════════════
#  TAB 2 — K-MEANS
# ══════════════════════════════════════════════
with tab_kmeans:
    st.markdown('<div class="section-tag">Optimal K</div>', unsafe_allow_html=True)
    st.markdown(f"#### Finding the Best Number of Clusters  —  Selected **k = {best_k}**")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor("#0d0d14")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#13131f")
        ax.grid(color="#1e1e30", lw=0.5)
        ax.tick_params(colors="#8b8b9e", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#2a2a4a")

    ax1.plot(ks, inertias, color="#c77dff", marker="o", markersize=6, lw=2, markerfacecolor="#7b2fff")
    ax1.axvline(best_k, color="#48cae4", lw=1.5, linestyle="--", alpha=0.8)
    ax1.set_title("Elbow — Inertia", color="#e8e4f0", fontsize=10)
    ax1.set_xlabel("k", color="#e8e4f0", fontsize=9)
    ax1.set_ylabel("Inertia", color="#e8e4f0", fontsize=9)

    bar_colors = ["#c77dff" if k == best_k else "#2a2a4a" for k in ks]
    ax2.bar(ks, silhouettes, color=bar_colors, edgecolor="#0d0d14", linewidth=0.5)
    ax2.set_title("Silhouette Score per k", color="#e8e4f0", fontsize=10)
    ax2.set_xlabel("k", color="#e8e4f0", fontsize=9)
    ax2.set_ylabel("Silhouette", color="#e8e4f0", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Metrics row
    mc1, mc2, mc3 = st.columns(3)
    for col, label, val, good in zip(
        [mc1, mc2, mc3],
        ["Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabász ↑"],
        [f"{k_sil:.3f}", f"{k_db:.3f}", f"{k_ch:.1f}"],
        [True, False, True]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{'#06d6a0' if good else '#f77f00'}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Cluster plot + summary side by side
    col_plot, col_table = st.columns([1.2, 1])

    with col_plot:
        st.markdown('<div class="section-tag">Cluster Map</div>', unsafe_allow_html=True)
        st.markdown("#### K-Means Clusters (PCA view)")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor("#0d0d14")
        ax.set_facecolor("#13131f")
        colors_used = CLUSTER_CMAP[:best_k]
        for cid, color in zip(range(best_k), colors_used):
            mask = k_labels == cid
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color,
                       s=50, alpha=0.85, edgecolor="#0d0d14", lw=0.3, label=f"Cluster {cid}")
        centers_pca = pca.transform(kmeans.cluster_centers_)
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                   c="white", s=180, marker="X", zorder=5, edgecolor="#0d0d14", lw=0.5, label="Centroids")
        ax.set_xlabel("PCA 1", color="#e8e4f0", fontsize=9)
        ax.set_ylabel("PCA 2", color="#e8e4f0", fontsize=9)
        ax.tick_params(colors="#8b8b9e", labelsize=8)
        ax.grid(color="#1e1e30", lw=0.5)
        for sp in ax.spines.values(): sp.set_color("#2a2a4a")
        ax.legend(fontsize=7, framealpha=0, labelcolor="#e8e4f0",
                  loc="upper right", fancybox=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_table:
        st.markdown('<div class="section-tag">Cluster Personas</div>', unsafe_allow_html=True)
        st.markdown("#### Segment Summary")
        for _, row in cluster_df.iterrows():
            lbl      = row["label"]
            info     = PERSONA_MAP.get(lbl, ("Unknown","#8b8b9e","#1a1a2e"))
            persona, color, bg = info
            cid_color = CLUSTER_CMAP[int(row["cluster"]) % len(CLUSTER_CMAP)]
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {color}44;
                        border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.6rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'Syne',sans-serif; font-weight:800;
                                 color:{cid_color}; font-size:0.85rem;">
                        Cluster {int(row['cluster'])}
                    </span>
                    <span style="background:{color}22; color:{color}; border:1px solid {color}55;
                                 border-radius:20px; padding:2px 10px; font-size:0.7rem; font-weight:700;">
                        {persona}
                    </span>
                </div>
                <div style="margin-top:0.5rem; font-size:0.8rem; color:#c0c0d0;">
                    👥 <b>{int(row['count'])}</b> customers &nbsp;|&nbsp;
                    💰 <b>${row['Annual Income (k$)']:.0f}k</b> &nbsp;|&nbsp;
                    🛒 <b>{row['Spending Score (1-100)']:.0f}</b>/100
                </div>
            </div>""", unsafe_allow_html=True)

    # Bar charts
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-tag">Bar Charts</div>', unsafe_allow_html=True)
    st.markdown("#### Mean Income & Spending per Cluster")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor("#0d0d14")
    bar_clrs = CLUSTER_CMAP[:best_k]
    for ax in [ax1, ax2]:
        ax.set_facecolor("#13131f")
        ax.tick_params(colors="#8b8b9e", labelsize=8)
        ax.grid(color="#1e1e30", lw=0.5, axis="y")
        for sp in ax.spines.values(): sp.set_color("#2a2a4a")

    ax1.bar(cluster_df["cluster"], cluster_df["Annual Income (k$)"], color=bar_clrs, edgecolor="#0d0d14")
    ax1.set_title("Mean Annual Income (k$)", color="#e8e4f0", fontsize=10)
    ax1.set_xlabel("Cluster", color="#e8e4f0", fontsize=9)

    ax2.bar(cluster_df["cluster"], cluster_df["Spending Score (1-100)"], color=bar_clrs, edgecolor="#0d0d14")
    ax2.set_title("Mean Spending Score", color="#e8e4f0", fontsize=10)
    ax2.set_xlabel("Cluster", color="#e8e4f0", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════
#  DBSCAN COMPUTATION
# ══════════════════════════════════════════════
db_model    = DBSCAN(eps=eps_val, min_samples=min_samp)
db_labels   = db_model.fit_predict(X_scaled)
df["DBSCAN_Cluster"] = db_labels
db_sil, db_db_score, db_ch = safe_metrics(X_scaled, db_labels)
n_db_clusters = len(set(db_labels) - {-1})
n_noise       = int((db_labels == -1).sum())


# ══════════════════════════════════════════════
#  TAB 3 — DBSCAN
# ══════════════════════════════════════════════
with tab_dbscan:
    if not run_dbscan:
        st.info("Enable DBSCAN in the sidebar to see results here.")
    else:
        st.markdown(f"#### DBSCAN  —  eps={eps_val}, min_samples={min_samp}")

        # Metrics
        d1, d2, d3, d4 = st.columns(4)
        for col, label, val in zip(
            [d1, d2, d3, d4],
            ["Clusters Found", "Noise Points", "Silhouette ↑", "Davies-Bouldin ↓"],
            [str(n_db_clusters), str(n_noise),
             f"{db_sil:.3f}" if not np.isnan(db_sil) else "N/A",
             f"{db_db_score:.3f}" if not np.isnan(db_db_score) else "N/A"]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value" style="font-size:1.6rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        col_p, col_g = st.columns([1.3, 1])

        with col_p:
            st.markdown("#### DBSCAN Clusters (PCA view)")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor("#0d0d14")
            ax.set_facecolor("#13131f")
            unique_labels = sorted(set(db_labels))
            for uid in unique_labels:
                mask  = db_labels == uid
                color = "#ef233c" if uid == -1 else CLUSTER_CMAP[uid % len(CLUSTER_CMAP)]
                label_str = "Noise" if uid == -1 else f"Cluster {uid}"
                marker = "x" if uid == -1 else "o"
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           color=color, s=45 if uid != -1 else 30,
                           alpha=0.8 if uid != -1 else 0.5,
                           marker=marker, edgecolor="#0d0d14" if uid != -1 else "none",
                           lw=0.3, label=label_str)
            ax.set_xlabel("PCA 1", color="#e8e4f0", fontsize=9)
            ax.set_ylabel("PCA 2", color="#e8e4f0", fontsize=9)
            ax.tick_params(colors="#8b8b9e", labelsize=8)
            ax.grid(color="#1e1e30", lw=0.5)
            for sp in ax.spines.values(): sp.set_color("#2a2a4a")
            ax.legend(fontsize=7, framealpha=0, labelcolor="#e8e4f0")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_g:
            st.markdown("#### Grid Search (eps × min_samples)")
            eps_list  = [0.3, 0.5, 0.8, 1.0, 1.2]
            ms_list   = [3, 5, 7, 10]
            grid_rows = []
            for e in eps_list:
                for ms in ms_list:
                    lbl = DBSCAN(eps=e, min_samples=ms).fit_predict(X_scaled)
                    s, _, _ = safe_metrics(X_scaled, lbl)
                    grid_rows.append({"eps": e, "min_samples": ms, "silhouette": s})
            grid_df = pd.DataFrame(grid_rows)
            pivot   = grid_df.pivot(index="min_samples", columns="eps", values="silhouette")

            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#0d0d14")
            ax.set_facecolor("#13131f")
            cmap2 = mcolors.LinearSegmentedColormap.from_list("purp", ["#13131f","#7b2fff","#c77dff"])
            sns.heatmap(pivot, ax=ax, cmap=cmap2, annot=True, fmt=".2f",
                        annot_kws={"size": 8, "color": "#e8e4f0"},
                        linecolor="#1a1a2e", linewidths=0.5,
                        cbar_kws={"shrink": 0.7})
            ax.set_title("Silhouette (NaN = invalid)", color="#e8e4f0", fontsize=9, pad=8)
            ax.tick_params(colors="#8b8b9e", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        if n_db_clusters > 0:
            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### DBSCAN Cluster Means (excluding noise)")
            db_sub = df[df["DBSCAN_Cluster"] != -1].groupby("DBSCAN_Cluster")[REQUIRED].mean().round(2)
            db_cnt = df["DBSCAN_Cluster"].value_counts().sort_index().rename("count")
            st.dataframe(pd.concat([db_cnt, db_sub], axis=1).style
                         .background_gradient(cmap="Purples", subset=REQUIRED)
                         .set_properties(**{"color": "white", "background-color": "#13131f"}),
                         use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — COMPARISON
# ══════════════════════════════════════════════
with tab_compare:
    st.markdown("#### K-Means vs DBSCAN — Metrics Comparison")

    comp_data = {
        "Algorithm":         ["K-Means",  "DBSCAN"],
        "Silhouette ↑":      [round(k_sil, 3), round(db_sil, 3) if not np.isnan(db_sil) else "N/A"],
        "Davies-Bouldin ↓":  [round(k_db, 3),  round(db_db_score, 3) if not np.isnan(db_db_score) else "N/A"],
        "Calinski-Harabász ↑":[round(k_ch, 1), round(db_ch, 1) if not np.isnan(db_ch) else "N/A"],
        "Clusters":          [best_k, n_db_clusters],
        "Noise Points":      [0, n_noise],
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df.set_index("Algorithm"), use_container_width=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Side-by-side scatter
    st.markdown("#### Side-by-Side Cluster Visualisation")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0d0d14")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#13131f")
        ax.tick_params(colors="#8b8b9e", labelsize=8)
        ax.grid(color="#1e1e30", lw=0.5)
        for sp in ax.spines.values(): sp.set_color("#2a2a4a")

    # K-Means
    for cid in range(best_k):
        mask = k_labels == cid
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=CLUSTER_CMAP[cid], s=45, alpha=0.85,
                    edgecolor="#0d0d14", lw=0.3, label=f"C{cid}")
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centers_pca[:, 0], centers_pca[:, 1],
                c="white", s=180, marker="X", zorder=5)
    ax1.set_title(f"K-Means  (k={best_k})", color="#e8e4f0", fontsize=10)
    ax1.legend(fontsize=7, framealpha=0, labelcolor="#e8e4f0")

    # DBSCAN
    for uid in sorted(set(db_labels)):
        mask  = db_labels == uid
        color = "#ef233c" if uid == -1 else CLUSTER_CMAP[uid % len(CLUSTER_CMAP)]
        lstr  = "Noise" if uid == -1 else f"C{uid}"
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=color, s=45 if uid != -1 else 25,
                    alpha=0.8 if uid != -1 else 0.4,
                    marker="o" if uid != -1 else "x",
                    edgecolor="#0d0d14" if uid != -1 else "none",
                    lw=0.3, label=lstr)
    ax2.set_title(f"DBSCAN  (eps={eps_val}, min_s={min_samp})", color="#e8e4f0", fontsize=10)
    ax2.legend(fontsize=7, framealpha=0, labelcolor="#e8e4f0")

    for ax in [ax1, ax2]:
        ax.set_xlabel("PCA 1", color="#e8e4f0", fontsize=9)
        ax.set_ylabel("PCA 2", color="#e8e4f0", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Winner banner
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    winner = "K-Means" if (np.isnan(db_sil) or k_sil > db_sil) else "DBSCAN"
    w_color = "#c77dff" if winner == "K-Means" else "#48cae4"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {w_color}18, {w_color}08);
                border: 1px solid {w_color}44; border-radius: 14px; padding: 1.2rem 1.8rem;">
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:{w_color};">
            🏆 Best Performer: {winner}
        </div>
        <div style="margin-top:0.4rem; font-size:0.85rem; color:#a0a0b4;">
            {'K-Means achieves cleaner, more compact clusters for this structured mall dataset. '
             'DBSCAN is better suited for irregular, density-varying shapes.' if winner=='K-Means'
             else 'DBSCAN outperforms K-Means on this dataset, detecting natural density-based structures.'}
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 5 — DATA
# ══════════════════════════════════════════════
with tab_data:
    st.markdown("#### Raw Dataset  with Cluster Labels")

    show_cols = list(df.columns)
    selected_cols = st.multiselect("Columns to display", show_cols, default=show_cols)

    search = st.text_input("🔍 Filter by cluster (e.g. enter 0, 1, 2 …)", "")
    display_df = df[selected_cols].copy()
    if search.strip():
        try:
            cid = int(search.strip())
            if "KMeans_Cluster" in display_df.columns:
                display_df = display_df[display_df["KMeans_Cluster"] == cid]
        except ValueError:
            pass

    st.dataframe(display_df, use_container_width=True, height=400)

    # Download
    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇  Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="segmented_customers.csv",
        mime="text/csv",
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Descriptive Statistics")
    st.dataframe(df[REQUIRED].describe().round(2), use_container_width=True)