import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# ════════════════════════════════════════════════════════════
# 1. KONFIGURASI GLOBAL
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="Retail Churn Intelligence", page_icon="🎯", layout="wide")

MODEL_DIR = "models/"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

REGIONAL_MAP = {
    'REGIONAL 1': 'SP',
    'REGIONAL 2': 'SMBR',
    'REGIONAL 6': 'ST'
}

# Menyembunyikan elemen bawaan Streamlit agar lebih bersih
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stMetric {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #0052cc;}
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cleansing_data(df):
    df = df.copy()
    for col in ['Tanggal Transaksi', 'Created_at_Dist']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col in ['Harga', 'TON Quantity', 'Harga_Per_KG']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    text_cols = ['Nama Toko', 'Area AP Toko', 'Brands', 'Cluster Pareto']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            
    if 'Area AP Toko' in df.columns:
        df['Area AP Toko'] = df['Area AP Toko'].map(REGIONAL_MAP).fillna(df['Area AP Toko'])
            
    df['Est_Spend'] = df['TON Quantity'] * df['Harga_Per_KG'].fillna(0)
    return df

@st.cache_data(show_spinner=False)
def feature_engineering(df, df_loyalty, cutoff):
    df_tr = df[df['Tanggal Transaksi'] <= cutoff].copy()
    if df_tr.empty: return pd.DataFrame()

    agg = df_tr.groupby('ID Toko').agg(
        last_trx=('Tanggal Transaksi', 'max'),
        Frequency=('No Transaksi', 'count'),
        Total_Ton=('TON Quantity', 'sum'),
        Monetary=('Est_Spend', 'sum'),
        Regional=('Area AP Toko', 'first'),
        Cluster_Pareto=('Cluster Pareto', 'first')
    )
    agg['Recency'] = (cutoff - agg['last_trx']).dt.days
    
    recent_ton = df_tr[df_tr['Tanggal Transaksi'] >= (cutoff - pd.Timedelta(days=30))].groupby('ID Toko')['TON Quantity'].sum()
    agg['Tonnage_Drop'] = (recent_ton / (agg['Total_Ton'] / 6 + 0.001)).fillna(0).clip(upper=10)
    
    if df_loyalty is not None and not df_loyalty.empty:
        col_id = df_loyalty.columns[0]
        lid = set(df_loyalty[col_id].astype(str).str.strip())
        agg['Is_Loyalty'] = np.where(agg.index.astype(str).isin(lid), 1, 0)
    else:
        agg['Is_Loyalty'] = 0
        
    return agg

def encode_features(agg, features_list=None):
    if agg.empty: return pd.DataFrame(), None
    enc = pd.get_dummies(agg, columns=['Regional'], prefix='Regional', dtype=int)
    enc.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in enc.columns]
    
    if features_list:
        enc = enc.reindex(columns=features_list, fill_value=0)
        
    drop_cols = ['last_trx', 'Label_Churn', 'Cluster_Pareto']
    feats = [c for c in enc.columns if c not in drop_cols]
    return enc[feats], (enc['Label_Churn'] if 'Label_Churn' in enc.columns else None)

# ════════════════════════════════════════════════════════════
# 3. INTERFACE & LOGIC
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2601/2601112.png", width=80)
    st.title("Data Input")
    uploaded_data = st.file_uploader("1. Transaksi (CSV/Parquet)", type=['parquet', 'csv'])
    uploaded_loyalty = st.file_uploader("2. Data Loyalty (Opsional)", type=['csv'])
    st.markdown("---")
    st.caption("Retail Churn Analytics v4.0")

def load_uploaded_file(file):
    if file is None: return None
    try:
        file.seek(0)
        return pd.read_parquet(file) if file.name.endswith('.parquet') else pd.read_csv(file)
    except Exception as e:
        st.error(f"Gagal membaca {file.name}")
        return None

df_raw = load_uploaded_file(uploaded_data)
df_l = load_uploaded_file(uploaded_loyalty)

try:
    model = joblib.load(os.path.join(MODEL_DIR, "model_best.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
    features_list = meta['features']
except:
    st.warning("⚠️ Model tidak ditemukan. Pastikan 'model_best.pkl' ada di folder 'models'.")
    model = None

# --- MAIN DASHBOARD ---
if df_raw is not None and model is not None:
    st.title("🎯 Retail Churn Intelligence")
    
    with st.spinner("Memproses miliaran byte data..."):
        df_clean = cleansing_data(df_raw)
        current_cutoff = df_clean['Tanggal Transaksi'].max()
        
        agg = feature_engineering(df_clean, df_l, current_cutoff)
        X_pred, _ = encode_features(agg, features_list)
        
        probs = model.predict_proba(X_pred)[:, 1]
        agg['Prob_Churn'] = probs
        agg['Level_Risiko'] = np.where(probs > 0.26, "🚨 TINGGI", "✅ AMAN")
        
        # Kalkulasi Metrik Bisnis
        total_toko = len(agg)
        toko_risiko = len(agg[agg['Level_Risiko'] == "🚨 TINGGI"])
        churn_rate = (toko_risiko / total_toko) * 100
        
        # Revenue at Risk (Jika kolom Harga tersedia)
        revenue_at_risk = agg[agg['Level_Risiko'] == "🚨 TINGGI"]['Monetary'].sum()

    st.markdown(f"**Data Historis Terakhir:** `{current_cutoff.strftime('%d %B %Y')}`")
    
    # 1. TOP METRICS (EXECUTIVE SUMMARY)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Toko Aktif", f"{total_toko:,}")
    c2.metric("Toko Berisiko (Prediksi)", f"{toko_risiko:,}", f"{churn_rate:.1f}% Churn Rate", delta_color="inverse")
    c3.metric("Loyalty Risk", len(agg[(agg['Level_Risiko'] == "🚨 TINGGI") & (agg['Is_Loyalty'] == 1)]))
    
    # Format Rupiah untuk Revenue at Risk
    if revenue_at_risk > 1e9:
        val_str = f"Rp {revenue_at_risk/1e9:.2f} Miliar"
    else:
        val_str = f"Rp {revenue_at_risk/1e6:.2f} Juta"
    c4.metric("Potensi Rupiah Hilang", val_str, "Revenue at Risk", delta_color="inverse")

    st.markdown("---")

    # 2. TABBED INTERFACE
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "📋 Actionable Watchlist", "🔍 Deep Dive Analytics"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            # Bar Chart: Risiko per Regional
            risk_by_region = agg[agg['Level_Risiko'] == "🚨 TINGGI"].groupby('Regional').size().reset_index(name='Jumlah Toko')
            fig_bar = px.bar(risk_by_region, x='Regional', y='Jumlah Toko', text_auto=True, 
                             title="Toko Berisiko Tinggi per Regional",
                             color='Regional', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_b:
            # Donut Chart: Komposisi Risiko
            risk_dist = agg['Level_Risiko'].value_counts().reset_index()
            risk_dist.columns = ['Status', 'Jumlah']
            fig_pie = px.pie(risk_dist, values='Jumlah', names='Status', hole=0.4, 
                             title="Persentase Status Risiko",
                             color='Status', color_discrete_map={"🚨 TINGGI": "#ef4444", "✅ AMAN": "#22c55e"})
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader("Daftar Toko Prioritas Intervensi")
        
        # Filter Dinamis
        f_col1, f_col2, f_col3 = st.columns(3)
        region_filter = f_col1.multiselect("Pilih Regional", options=agg['Regional'].unique(), default=agg['Regional'].unique())
        loyalty_filter = f_col2.selectbox("Status Loyalty", options=["Semua", "Hanya Peserta Loyalty", "Non-Loyalty"])
        
        display_df = agg[agg['Regional'].isin(region_filter)].copy()
        if loyalty_filter == "Hanya Peserta Loyalty":
            display_df = display_df[display_df['Is_Loyalty'] == 1]
        elif loyalty_filter == "Non-Loyalty":
            display_df = display_df[display_df['Is_Loyalty'] == 0]
            
        display_df = display_df.sort_values('Prob_Churn', ascending=False)
        
        # Merapikan tabel untuk dilihat
        tabel_tayang = display_df[['Regional', 'Cluster_Pareto', 'Recency', 'Tonnage_Drop', 'Prob_Churn', 'Level_Risiko']].copy()
        tabel_tayang['Prob_Churn'] = (tabel_tayang['Prob_Churn'] * 100).round(1).astype(str) + "%"
        tabel_tayang['Tonnage_Drop'] = tabel_tayang['Tonnage_Drop'].round(2)
        
        st.dataframe(tabel_tayang, use_container_width=True, height=400)
        st.download_button("📥 Ekspor Data ke CSV", display_df.to_csv().encode('utf-8'), "watchlist_eksekusi.csv")

    with tab3:
        st.subheader("Analisis Perilaku Pelanggan")
        col_c, col_d = st.columns(2)
        
        with col_c:
            # Scatter Plot: Recency vs Tonnage Drop
            fig_scatter = px.scatter(agg, x="Recency", y="Tonnage_Drop", color="Level_Risiko", 
                                     size="Monetary", hover_name=agg.index, opacity=0.7,
                                     title="Peta Risiko: Recency vs Tonnage Drop",
                                     color_discrete_map={"🚨 TINGGI": "red", "✅ AMAN": "green"})
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_d:
            # Histogram: Distribusi Probabilitas
            fig_hist = px.histogram(agg, x="Prob_Churn", nbins=20, 
                                    title="Distribusi Probabilitas Churn",
                                    labels={'Prob_Churn': 'Probabilitas Churn', 'count': 'Jumlah Toko'},
                                    color_discrete_sequence=['#636efa'])
            fig_hist.add_vline(x=0.26, line_dash="dash", line_color="red", annotation_text="Batas Risiko (26%)")
            st.plotly_chart(fig_hist, use_container_width=True)

else:
    # Tampilan awal jika belum ada data yang diunggah
    st.markdown("<h1 style='text-align: center; color: #888;'>Selamat Datang di Retail Churn Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Silakan unggah data transaksi Anda pada panel di sebelah kiri untuk memulai analisis.</p>", unsafe_allow_html=True)
