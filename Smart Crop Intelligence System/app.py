"""
app.py
------
Streamlit Web Application for the
Crop Yield Prediction & Crop Recommendation System.

Tabs:
  🏠 Overview
  🔍 Exploratory Data Analysis
  🌾 Crop Yield Prediction
  💡 Crop Recommendation
  📊 Model Analytics
  ℹ️  About
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── project root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import (
    recommend_crops, CROP_REQUIREMENTS, MODEL_DIR, VIZ_DIR,
    load_dataset, logger,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Intelligence System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Gradient hero */
.hero {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 40%, #52b788 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.hero h1 { font-size: 2.8rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.9; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border-left: 4px solid #52b788;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card h3 { color: #2d6a4f; font-size: 1.9rem; margin: 0; font-weight: 700; }
.metric-card p  { color: #6c757d; font-size: 0.82rem; margin: 0.3rem 0 0 0; text-transform: uppercase; letter-spacing: 0.5px; }

/* Recommendation cards */
.rec-card {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #74c69d;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(82,183,136,0.2);
}
.rec-card h2 { color: #1b4332; font-size: 1.5rem; margin: 0; }
.rec-card .score { color: #2d6a4f; font-size: 2rem; font-weight: 700; }
.rec-card .label { font-size: 0.78rem; color: #6c757d; text-transform: uppercase; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #2d6a4f, #52b788);
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    margin: 1.5rem 0 1rem 0;
}

/* Prediction result */
.pred-result {
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(27,67,50,0.3);
}
.pred-result h1 { font-size: 3.5rem; margin: 0; font-weight: 700; }
.pred-result p  { font-size: 1rem; margin: 0.3rem 0 0 0; opacity: 0.85; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a472a 0%, #2d6a4f 100%) !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSlider > label { color: white !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return load_dataset()
    except FileNotFoundError:
        st.warning("Dataset not found. Run `python data/generate_dataset.py` first.")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_trained_models():
    """Load pre-trained model artefacts if they exist."""
    artefacts = {}
    files = {
        "model":    "best_model.pkl",
        "encoder":  "encoder.pkl",
        "scaler":   "scaler.pkl",
        "features": "feature_cols.pkl",
    }
    for key, fname in files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            artefacts[key] = joblib.load(path)
    return artefacts


def try_load_results():
    """Try to load saved model comparison results."""
    path = os.path.join(MODEL_DIR, "results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌾 Smart Crop Intelligence System")
    st.markdown("---")
    st.markdown("**Research-Grade System**")
    st.markdown("Predict crop yield and get smart recommendations using advanced ML models.")
    st.markdown("---")
    st.markdown("### 🧪 Model Pipeline")
    st.markdown("- ✅ Random Forest")
    st.markdown("- ✅ XGBoost")
    st.markdown("- ✅ LightGBM")
    st.markdown("- ✅ CatBoost")
    st.markdown("- ✅ Gradient Boosting")
    st.markdown("- ✅ SVM")
    st.markdown("- ✅ Neural Network")
    st.markdown("- ✅ Stacking Ensemble")
    st.markdown("---")
    st.markdown("### ⚡ Quick Guide")
    st.markdown("1. Use **Prediction** tab to get yield estimates")
    st.markdown("2. Use **Recommendation** tab for crop suggestions")
    st.markdown("3. Explore **Analytics** for model insights")
    st.markdown("---")
    st.caption("v2.0 · MAWT-SVM Improved System")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

df = load_data()
artefacts = load_trained_models()

tab_overview, tab_eda, tab_predict, tab_recommend, tab_analytics, tab_about = st.tabs([
    "🏠 Overview",
    "🔍 EDA",
    "🌾 Yield Prediction",
    "💡 Crop Recommendation",
    "📊 Model Analytics",
    "ℹ️ About",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown("""
    <div class="hero">
        <h1>🌾 Crop Intelligence System</h1>
        <p>Advanced Crop Yield Prediction & Recommendation using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <h3>{len(df):,}</h3><p>Total Records</p></div>""", unsafe_allow_html=True)
        with col2:
            n_crops = df["Crop"].nunique() if "Crop" in df.columns else "—"
            st.markdown(f"""<div class="metric-card">
                <h3>{n_crops}</h3><p>Crop Varieties</p></div>""", unsafe_allow_html=True)
        with col3:
            n_states = df["State"].nunique() if "State" in df.columns else "—"
            st.markdown(f"""<div class="metric-card">
                <h3>{n_states}</h3><p>States Covered</p></div>""", unsafe_allow_html=True)
        with col4:
            avg_yield = f"{df['Yield'].mean():.2f}" if "Yield" in df.columns else "—"
            st.markdown(f"""<div class="metric-card">
                <h3>{avg_yield}</h3><p>Avg Yield (t/ha)</p></div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_left, col_right = st.columns([1.2, 0.8])
        with col_left:
            st.markdown('<div class="section-header">📈 Yield by Crop</div>', unsafe_allow_html=True)
            if "Crop" in df.columns and "Yield" in df.columns:
                crop_yield = df.groupby("Crop")["Yield"].agg(["mean", "std"]).reset_index()
                crop_yield.columns = ["Crop", "Mean Yield", "Std"]
                crop_yield = crop_yield.sort_values("Mean Yield", ascending=True)
                fig = px.bar(crop_yield, x="Mean Yield", y="Crop",
                             orientation="h", error_x="Std",
                             color="Mean Yield",
                             color_continuous_scale="Greens",
                             template="plotly_white")
                fig.update_layout(height=420, showlegend=False,
                                  coloraxis_showscale=False,
                                  margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">🗓️ Season Distribution</div>', unsafe_allow_html=True)
            if "Season" in df.columns:
                season_counts = df["Season"].value_counts().reset_index()
                season_counts.columns = ["Season", "Count"]
                fig2 = px.pie(season_counts, names="Season", values="Count",
                              color_discrete_sequence=px.colors.sequential.Greens[-4:],
                              template="plotly_white", hole=0.4)
                fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("No data found. Please generate the dataset first: `python data/generate_dataset.py`")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ═════════════════════════════════════════════════════════════════════════════

with tab_eda:
    st.title("🔍 Exploratory Data Analysis")

    if df.empty:
        st.info("No data loaded.")
    else:
        # Summary stats
        st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            corr_cols = st.multiselect("Select features for correlation",
                                        num_cols, default=num_cols[:8])
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr()
                fig_corr = px.imshow(corr, text_auto=".2f",
                                     color_continuous_scale="RdYlGn",
                                     aspect="auto", template="plotly_white")
                fig_corr.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">📦 Distribution Explorer</div>', unsafe_allow_html=True)
            sel_col = st.selectbox("Select feature", num_cols)
            group_col = None
            if "Crop" in df.columns:
                group_col = st.selectbox("Colour by", ["None", "Crop", "Season"])
                if group_col == "None":
                    group_col = None

            fig_hist = px.histogram(df, x=sel_col, color=group_col,
                                    nbins=40, template="plotly_white",
                                    color_discrete_sequence=px.colors.qualitative.Safe,
                                    marginal="box")
            fig_hist.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">🌱 Yield vs Feature Scatter</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            x_feat = st.selectbox("X-axis feature", num_cols,
                                   index=num_cols.index("Rainfall") if "Rainfall" in num_cols else 0)
        with c2:
            hue = st.selectbox("Colour by (scatter)", ["None", "Crop", "Season"])
            hue = None if hue == "None" else hue

        if "Yield" in df.columns:
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig_scat = px.scatter(sample, x=x_feat, y="Yield",
                                   color=hue,
                                   opacity=0.5, template="plotly_white",
                                   color_discrete_sequence=px.colors.qualitative.Safe)
            fig_scat.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=30))
            st.plotly_chart(fig_scat, use_container_width=True)

        # Missing value heatmap
        if df.isnull().sum().sum() > 0:
            st.markdown('<div class="section-header">❓ Missing Values</div>', unsafe_allow_html=True)
            miss = df.isnull().sum().reset_index()
            miss.columns = ["Feature", "Missing"]
            miss = miss[miss["Missing"] > 0].sort_values("Missing", ascending=False)
            fig_miss = px.bar(miss, x="Feature", y="Missing",
                              color="Missing", color_continuous_scale="Reds",
                              template="plotly_white")
            st.plotly_chart(fig_miss, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – YIELD PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.title("🌾 Crop Yield Prediction")

    model_loaded = "model" in artefacts and "scaler" in artefacts and "features" in artefacts

    if not model_loaded:
        st.warning("⚠️ No trained model found. Please run the training pipeline first:")
        st.code("python train_model.py --skip-hpo", language="bash")
        st.info("The system will use a rule-of-thumb estimator until a model is trained.")

    st.markdown("### Enter Field Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌿 Soil Nutrients**")
        N  = st.slider("Nitrogen (N) [kg/ha]", 0.0, 300.0, 90.0, 0.5)
        P  = st.slider("Phosphorus (P) [kg/ha]", 0.0, 200.0, 45.0, 0.5)
        K  = st.slider("Potassium (K) [kg/ha]", 0.0, 250.0, 43.0, 0.5)
        pH = st.slider("Soil pH", 3.5, 9.5, 6.5, 0.05)

    with col2:
        st.markdown("**🌦️ Weather**")
        temp     = st.slider("Temperature (°C)", -10.0, 55.0, 25.0, 0.1)
        rainfall = st.slider("Rainfall (mm/yr)", 0.0, 5000.0, 900.0, 10.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 0.5)
        solar    = st.slider("Solar Radiation (MJ/m²)", 5.0, 30.0, 18.0, 0.1)

    with col3:
        st.markdown("**🌾 Crop & Field**")
        crop_list = sorted(CROP_REQUIREMENTS.keys())
        crop      = st.selectbox("Crop", crop_list)
        area      = st.number_input("Area (ha)", min_value=1.0, max_value=500_000.0, value=1000.0)
        pesticide = st.slider("Pesticide Usage (kg/ha)", 0.0, 50.0, 5.0, 0.1)
        season    = st.selectbox("Season", ["Kharif", "Rabi", "Zaid", "Whole Year"])

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Yield", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Running prediction model …"):
            if model_loaded:
                # Build input using the saved feature list
                features = artefacts["features"]
                encoder  = artefacts.get("encoder")
                scaler   = artefacts["scaler"]

                # Map crop to encoded value
                crop_encoded = 0
                if encoder and "Crop" in encoder.label_encoders:
                    try:
                        crop_encoded = encoder.label_encoders["Crop"].transform([crop])[0]
                    except Exception:
                        crop_encoded = 0

                # Season OHE columns
                season_map = {f"Season_{s}": 0 for s in ["Kharif", "Rabi", "Whole Year", "Zaid"]}
                season_key = f"Season_{season}"
                if season_key in season_map:
                    season_map[season_key] = 1

                input_dict = {
                    "Temperature":    temp,
                    "Rainfall":       rainfall,
                    "Humidity":       humidity,
                    "pH":             pH,
                    "N":              N,
                    "P":              P,
                    "K":              K,
                    "Pesticide_usage":pesticide,
                    "Area":           area,
                    "Year":           2024.0,
                    "Crop":           crop_encoded,
                    "solar_radiation":solar,
                    "wind_speed":     3.0,
                    "soil_moisture":  45.0,
                    **season_map,
                }

                # Build DataFrame with feature columns
                row = pd.DataFrame([{f: input_dict.get(f, 0.0) for f in features}])

                try:
                    row_scaled = scaler.transform(row)
                    pred_yield = float(artefacts["model"].predict(
                        row_scaled.values if hasattr(row_scaled, "values") else row_scaled
                    )[0])
                except Exception:
                    try:
                        pred_yield = float(artefacts["model"].predict(row.values)[0])
                    except Exception as e:
                        pred_yield = None
                        st.error(f"Prediction error: {e}")
            else:
                # Fallback rule-of-thumb estimate
                score = (
                    0.25 * min(N / 120, 1.5) +
                    0.20 * min(P / 60, 1.5) +
                    0.20 * min(K / 50, 1.5) +
                    0.20 * min(rainfall / 1000, 2.0) +
                    0.10 * (1 - abs(temp - 27) / 20) +
                    0.05 * min(humidity / 70, 1.5)
                ) * (1 - pesticide / 200)
                pred_yield = round(score * 8, 2)

            if pred_yield is not None:
                production = pred_yield * area

                # Display result
                category = "Low 🟡" if pred_yield < 5 else ("Medium 🟠" if pred_yield < 10 else "High 🟢")
                st.markdown(f"""
                <div class="pred-result">
                    <p>Estimated Crop Yield</p>
                    <h1>{pred_yield:.2f} <small style='font-size:1.5rem'>t/ha</small></h1>
                    <p>Category: <strong>{category}</strong> &nbsp;|&nbsp; Total Production: <strong>{production:,.0f} tonnes</strong></p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_yield,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Yield (t/ha)", "font": {"size": 18}},
                    delta={"reference": 7.0, "increasing": {"color": "#52b788"}},
                    gauge={
                        "axis":  {"range": [0, 20], "tickwidth": 1},
                        "bar":   {"color": "#2d6a4f"},
                        "steps": [
                            {"range": [0,  5],  "color": "#fed9a6"},
                            {"range": [5,  10], "color": "#a8dadc"},
                            {"range": [10, 20], "color": "#b7e4c7"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3},
                                      "thickness": 0.75, "value": 7},
                    }
                ))
                fig_gauge.update_layout(height=300)
                col_g1, col_g2 = st.columns([1, 1])
                with col_g1:
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with col_g2:
                    st.markdown("### 📊 Quick Insights")
                    st.info(f"**Crop selected:** {crop}")
                    st.info(f"**Predicted production:** {production:,.1f} tonnes over {area:,.0f} ha")
                    if pred_yield < 5:
                        st.warning("🔔 Low yield detected. Consider optimizing N/P/K ratios and irrigation.")
                    elif pred_yield < 10:
                        st.success("✅ Moderate yield. Good agronomic practices are in place.")
                    else:
                        st.success("🌟 Excellent yield conditions! Continue these practices.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 – CROP RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════════════

with tab_recommend:
    st.title("💡 Smart Crop Recommendation")
    st.markdown("Enter your soil and weather conditions to get the **Top 3 recommended crops**.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌿 Soil Profile**")
        r_N    = st.slider("Nitrogen (N)", 0.0, 300.0, 90.0, key="rN")
        r_P    = st.slider("Phosphorus (P)", 0.0, 200.0, 42.0, key="rP")
        r_K    = st.slider("Potassium (K)", 0.0, 250.0, 43.0, key="rK")
        r_pH   = st.slider("Soil pH", 3.5, 9.5, 6.5, 0.05, key="rpH")

    with col2:
        st.markdown("**🌦️ Climate**")
        r_temp = st.slider("Temperature (°C)", -10.0, 55.0, 25.0, 0.1, key="rtemp")
        r_rain = st.slider("Rainfall (mm/yr)", 0.0, 5000.0, 900.0, 10.0, key="rrain")
        r_hum  = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 0.5, key="rhum")

    rec_btn = st.button("🌱 Get Recommendations", type="primary", use_container_width=True)

    if rec_btn:
        with st.spinner("Analysing crop suitability …"):
            recommendations = recommend_crops(
                N=r_N, P=r_P, K=r_K,
                temperature=r_temp, rainfall=r_rain, humidity=r_hum,
                pH=r_pH, top_n=3,
            )

        st.markdown("### 🏆 Top 3 Recommended Crops")
        medals = ["🥇", "🥈", "🥉"]
        medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

        rec_cols = st.columns(3)
        for i, (col, rec) in enumerate(zip(rec_cols, recommendations)):
            with col:
                score_pct = float(rec["suitability"].replace("%", ""))
                st.markdown(f"""
                <div class="rec-card">
                    <div style="font-size:2rem">{medals[i]}</div>
                    <h2>{rec['crop']}</h2>
                    <div class="score">{rec['suitability']}</div>
                    <div class="label">Suitability Score</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Radar chart: suitability across all crops
        st.markdown("### 📡 Suitability Radar")
        all_recs = recommend_crops(r_N, r_P, r_K, r_temp, r_rain, r_hum, r_pH, top_n=10)
        rec_df = pd.DataFrame(all_recs)
        rec_df["score_pct"] = rec_df["suitability"].str.replace("%", "").astype(float)

        fig_bar = px.bar(rec_df, x="crop", y="score_pct",
                          color="score_pct",
                          color_continuous_scale="Greens",
                          labels={"crop": "Crop", "score_pct": "Suitability (%)"},
                          template="plotly_white")
        fig_bar.update_layout(height=380, showlegend=False,
                               coloraxis_showscale=False,
                               xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Requirements comparison table
        st.markdown("### 📋 Ideal Ranges for Top 3 Crops")
        rows = []
        for rec in recommendations:
            req = CROP_REQUIREMENTS.get(rec["crop"], {})
            rows.append({
                "Crop":        rec["crop"],
                "Suitability": rec["suitability"],
                "N (kg/ha)":   f"{req.get('N', (0,0))[0]} – {req.get('N', (0,0))[1]}",
                "P (kg/ha)":   f"{req.get('P', (0,0))[0]} – {req.get('P', (0,0))[1]}",
                "K (kg/ha)":   f"{req.get('K', (0,0))[0]} – {req.get('K', (0,0))[1]}",
                "Temp (°C)":   f"{req.get('temp', (0,0))[0]} – {req.get('temp', (0,0))[1]}",
                "Rain (mm)":   f"{req.get('rain', (0,0))[0]} – {req.get('rain', (0,0))[1]}",
                "pH":          f"{req.get('pH', (0,0))[0]} – {req.get('pH', (0,0))[1]}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 – MODEL ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.title("📊 Model Analytics & Insights")

    # Check for saved visualisations
    viz_files = {
        "model_comparison.png":    "Model Comparison",
        "feature_importance.png":  "Feature Importance",
        "actual_vs_predicted.png": "Actual vs Predicted",
        "residuals.png":           "Residual Analysis",
        "correlation_heatmap.png": "Correlation Heatmap",
        "yield_distribution.png":  "Yield Distribution",
        "shap_importance.png":     "SHAP Feature Importance",
    }

    found = {label: os.path.join(VIZ_DIR, fname)
             for fname, label in viz_files.items()
             if os.path.exists(os.path.join(VIZ_DIR, fname))}

    if not found:
        st.info("No analytics images found yet. Train the models first:")
        st.code("python train_model.py --skip-hpo", language="bash")
        st.markdown("---")
        st.markdown("### 🔬 What you'll see after training:")
        cols = st.columns(3)
        items = [
            ("📈 Model Comparison", "RMSE, MAE, R² charts for all models"),
            ("🌳 Feature Importance", "Which soil/weather factors matter most"),
            ("⭕ SHAP Values", "Explainable AI — why each prediction was made"),
            ("📉 Residual Analysis", "Error distribution & diagnostic plots"),
            ("🎯 Actual vs Predicted", "Scatter plot of prediction accuracy"),
            ("🔥 Correlation Heatmap", "Feature inter-relationships"),
        ]
        for i, (title, desc) in enumerate(items):
            with cols[i % 3]:
                st.markdown(f"**{title}**\n\n{desc}")
    else:
        # Grid layout for images
        labels = list(found.keys())
        paths  = list(found.values())

        for i in range(0, len(labels), 2):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{labels[i]}**")
                st.image(paths[i], use_container_width=True)
            if i + 1 < len(labels):
                with c2:
                    st.markdown(f"**{labels[i+1]}**")
                    st.image(paths[i+1], use_container_width=True)
            st.markdown("---")

    # Model results table
    results = try_load_results()
    if results is not None:
        st.markdown("### 📋 Model Performance Table")
        st.dataframe(results.style.highlight_max(subset=["R²"], color="#b7e4c7")
                               .highlight_min(subset=["RMSE", "MAE"], color="#b7e4c7"),
                     use_container_width=True)

    # Live exploratory plots (from loaded data)
    if not df.empty and "Yield" in df.columns:
        st.markdown("---")
        st.markdown("### 🌡️ Interactive Yield Explorer")
        col1, col2 = st.columns(2)
        with col1:
            feat1 = st.selectbox("X-axis", [c for c in df.select_dtypes(include=np.number).columns if c != "Yield"], key="a1")
        with col2:
            feat2 = st.selectbox("Y-axis", [c for c in df.select_dtypes(include=np.number).columns if c != "Yield"], index=1, key="a2")

        sample = df.sample(min(3000, len(df)), random_state=42)
        hue_col = "Crop" if "Crop" in sample.columns else None
        fig_exp = px.scatter(sample, x=feat1, y=feat2,
                              color=hue_col, size="Yield" if "Yield" in sample.columns else None,
                              opacity=0.6, template="plotly_white",
                              color_discrete_sequence=px.colors.qualitative.Safe)
        fig_exp.update_layout(height=430)
        st.plotly_chart(fig_exp, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 – ABOUT
# ═════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.title("ℹ️ About This System")

    st.markdown("""
    ## 🌾 Crop Intelligence System — v2.0

    This research-grade system improves upon the base paper
    **"Crop Yield Prediction using Multi-Attribute Weighted Tree-Based Support Vector Machine (MAWT-SVM)"**
    by developing a comprehensive hybrid ML framework.

    ---

    ## 🧠 Technical Architecture

    | Layer | Component | Details |
    |-------|-----------|---------|
    | **Data** | Multi-source Dataset | 8,000+ records, 20 crops, 15 states |
    | **Preprocessing** | Full Pipeline | Imputation, encoding, scaling, outlier removal |
    | **Features** | Selection Methods | PCA, RFE, Mutual Info, SHAP |
    | **Models** | ML Ensemble | RF, SVM, XGBoost, LightGBM, CatBoost, GBM, MLP |

    | **Ensemble** | Stacking | XGB + GBM + RF → Ridge meta-learner |
    | **HPO** | Optimization | GridSearchCV, RandomizedSearch, Optuna |
    | **XAI** | Explainability | SHAP values + Partial Dependence Plots |
    | **UI** | Web App | Streamlit with interactive Plotly charts |

    ---

    ## 🌱 Supported Crops (20)

    Rice, Wheat, Maize, Cotton, Sugarcane, Soybean, Groundnut, Barley,
    Chickpea, Mustard, Potato, Tomato, Onion, Jowar, Bajra, Sunflower,
    Turmeric, Ginger, Tea, Coffee

    ---

    ## 📐 Feature Set

    | Category | Features |
    |----------|----------|
    | **Soil Nutrients** | N, P, K, pH, Soil Moisture |
    | **Weather** | Temperature, Rainfall, Humidity, Solar Radiation, Wind Speed |
    | **Crop Info** | Crop Name, Season, Area, Year |
    | **Location** | State, District |
    | **Agronomy** | Pesticide Usage, Production |

    ---

    ## 🚀 Quick Start

    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Generate dataset
    python data/generate_dataset.py

    # 3. Train models (quick mode)
    python train_model.py --skip-hpo

    # 4. Full training with HPO
    python train_model.py

    # 5. Launch app
    streamlit run app.py
    ```

    ---

    ## 📊 Evaluation Metrics

    The system evaluates all models using **RMSE, MAE, R², MAPE, NRMSE**
    for regression (yield prediction), and **Accuracy, Precision, Recall, F1**
    for crop classification.

    ---
    *Developed as an advanced research project extending state-of-the-art MAWT-SVM approach.*
    """)
