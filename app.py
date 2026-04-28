import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import plotly.express as px

# ════════════════════════════════════════════════════════════
# 1. KONFIGURASI GLOBAL & STYLE
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="Retail Churn Intelligence v4.2", page_icon="🎯", layout="wide")

MODEL_DIR = "models/"
REGIONAL_MAP = {'REGIONAL 1': 'SP', 'REGIONAL 2': 'SMBR', 'REGIONAL 6': 'ST'}

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stMetric {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #0052cc;}
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS (SINKRON DENGAN COLAB TERBARU)
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cleansing_data(df):
    df = df.copy()
    # Konversi Tanggal
    for col in ['Tanggal Transaksi', 'Created_at_Dist']:
        if col in df.columns: 
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Konversi Numerik
    for col in ['Harga', 'Zak Quantity', 'TON Quantity', 'Harga_Per_KG']:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Normalisasi Teks
    text_cols = ['Nama Toko', 'Area AP Toko', 'Brands', 'Cluster Pareto', 'Tipe Customer', 'SSM', 'ASM', 'TSO']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().replace({'NAN':np.nan, 'NONE':np.nan, '':np.nan})
            
    if 'Area AP Toko' in df.columns: 
        df['Area AP Toko'] = df['Area AP Toko'].map(REGIONAL_MAP).fillna(df['Area AP Toko'])
        
    # Fillna Cerdas
    if 'Cluster Pareto' in df.columns: df['Cluster Pareto'] = df['Cluster Pareto'].fillna('BRONZE')
    for col in ['SSM', 'ASM', 'TSO', 'Area AP Toko', 'Tipe Customer', 'Brands']:
        if col in df.columns: df[col] = df[col].fillna('UNKNOWN')

    if 'TON Quantity' in df.columns and 'Harga_Per_KG' in df.columns:
        df['Est_Spend'] = df['TON Quantity'] * 1000 * df['Harga_Per_KG'].fillna(0)
        
    return df

@st.cache_data(show_spinner=False)
def feature_engineering(df, df_loyalty, cutoff):
    RECENT_DAYS = 30
    PAST_DAYS = 120

    df_tr = df[df['Tanggal Transaksi'] <= cutoff].copy()
    if df_tr.empty: return pd.DataFrame()
    
    df_tr['Days_Ago'] = (cutoff - df_tr['Tanggal Transaksi']).dt.days

    # AGREGASI 143 FITUR (RFM, Salesman, Brand, dll)
    agg = df_tr.groupby('ID Toko').agg(
        last_trx         = ('Tanggal Transaksi', 'max'),
        Frequency        = ('No Transaksi', 'count'),
        Total_Ton        = ('TON Quantity', 'sum'),
        Monetary         = ('Est_Spend', 'sum'),
        Avg_Harga_Per_KG = ('Harga_Per_KG', 'mean'),
        Regional         = ('Area AP Toko', 'first'),
        Cluster_Pareto   = ('Cluster Pareto', 'first'),
        Tipe_Customer    = ('Tipe Customer', 'first'),
        Dominant_Brand   = ('Brands', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        Dominant_SSM     = ('SSM', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        Dominant_ASM     = ('ASM', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        Dominant_TSO     = ('TSO', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
    )
    
    agg['Recency'] = (cutoff - agg['last_trx']).dt.days

    # Tonnage Drop logic
    recent = df_tr[df_tr['Days_Ago'] <= RECENT_DAYS].groupby('ID Toko')['TON Quantity'].sum()
    past = (df_tr[(df_tr['Days_Ago'] > RECENT_DAYS) & (df_tr['Days_Ago'] <= PAST_DAYS)]
            .groupby('ID Toko')['TON Quantity'].sum() / 3)
    agg['Tonnage_Drop'] = (recent / (past + 0.001)).fillna(0).clip(upper=10)

    # RFM Scoring sederhana untuk visualisasi
    agg['R_Score'] = pd.qcut(agg['Recency'], q=5, labels=False, duplicates='drop').fillna(0)
    agg['F_Score'] = pd.qcut(agg['Frequency'], q=5, labels=False, duplicates='drop').fillna(0)
    agg['M_Score'] = pd.qcut(agg['Monetary'], q=5, labels=False, duplicates='drop').fillna(0)
    agg['RFM_Score'] = (agg['R_Score'] + agg['F_Score'] + agg['M_Score']) / 3

    if df_loyalty is not None and not df_loyalty.empty:
        col_id = df_loyalty.columns[0]
        lid = set(df_loyalty[col_id].astype(str).str.strip())
        agg['Is_Loyalty'] = np.where(agg.index.astype(str).isin(lid), 1, 0)
    else:
        agg['Is_Loyalty'] = 0

    return agg

def encode_features(agg, features_list):
    if agg.empty: return pd.DataFrame()
    
    # One-Hot Encoding semua kolom kategori
    enc_cols = ['Regional', 'Dominant_Brand', 'Tipe_Customer', 'Dominant_SSM', 'Dominant_ASM', 'Dominant_TSO']
    enc_cols = [c for c in enc_cols if c in agg.columns]
    
    enc = pd.get_dummies(agg, columns=enc_cols, prefix=enc_cols, dtype=int)
    enc.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in enc.columns]
    
    # SINKRONISASI PAKSA: Memastikan kolom 100% cocok dengan metadata model Colab
    enc = enc.reindex(columns=features_list, fill_value=0)
    return enc[features_list]

# ════════════════════════════════════════════════════════════
# 3. SIDEBAR & DATA LOADING
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📂 Input Data")
    st.info("💡 Unggah file transaksi gabungan (2025 & 2026) untuk akurasi terbaik.")
    uploaded_data = st.file_uploader("1. Transaksi (Parquet/CSV)", type=['parquet', 'csv'])
    uploaded_loyalty = st.file_uploader("2. Data Loyalty (CSV)", type=['csv'])

# Load Model & Metadata
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model_best.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
    features_list = meta['features']
except:
    st.error("🚨 Gagal memuat model. Pastikan file .pkl dan .json ada di folder 'models/'.")
    st.stop()

# ════════════════════════════════════════════════════════════
# 4. DASHBOARD EXECUTION
# ════════════════════════════════════════════════════════════

if uploaded_data:
    # Membaca data
    uploaded_data.seek(0)
    df_raw = pd.read_parquet(uploaded_data) if uploaded_data.name.endswith('.parquet') else pd.read_csv(uploaded_data)
    df_l = pd.read_csv(uploaded_loyalty) if uploaded_loyalty else None

    st.title("🎯 Retail Churn Intelligence")
    
    with st.spinner("Menganalisis 143 parameter perilaku pelanggan..."):
        df_clean = cleansing_data(df_raw)
        current_cutoff = df_clean['Tanggal Transaksi'].max()
        
        # Ekstraksi Fitur & Prediksi
        agg = feature_engineering(df_clean, df_l, current_cutoff)
        X_pred = encode_features(agg, features_list)
        
        probs = model.predict_proba(X_pred)[:, 1]
        agg['Prob_Churn'] = probs
        agg['Level_Risiko'] = np.where(probs > 0.26, "🚨 TINGGI", "✅ AMAN")
        
        # Metrik
        t_risk = len(agg[agg['Level_Risiko'] == "🚨 TINGGI"])
        rev_risk = agg[agg['Level_Risiko'] == "🚨 TINGGI"]['Monetary'].sum()

    # Layout Dashboard
    st.write(f"📅 **Data Terbaru:** {current_cutoff.strftime('%d %B %Y')}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Toko", f"{len(agg):,}")
    m2.metric("Risiko Churn", f"{t_risk:,}", f"{(t_risk/len(agg)*100):.1f}%")
    m3.metric("Loyalty Risk", len(agg[(agg['Level_Risiko'] == "🚨 TINGGI") & (agg['Is_Loyalty'] == 1)]))
    m4.metric("Revenue at Risk", f"Rp {rev_risk/1e6:.1f} Juta")

    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📋 Watchlist Prioritas", "📊 Analisis Wilayah"])
    
    with tab1:
        st.subheader("Daftar Toko Berisiko (Segera Intervensi)")
        # Filter Regional
        regs = st.multiselect("Filter Regional", options=agg['Regional'].unique(), default=agg['Regional'].unique())
        df_view = agg[agg['Regional'].isin(regs)].sort_values('Prob_Churn', ascending=False)
        
        st.dataframe(df_view[['Regional', 'Cluster_Pareto', 'Recency', 'Tonnage_Drop', 'Prob_Churn', 'Level_Risiko']], use_container_width=True)
        st.download_button("📥 Ekspor Watchlist", df_view.to_csv().encode('utf-8'), "watchlist_churn.csv")

    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            fig_bar = px.bar(agg[agg['Level_Risiko']=="🚨 TINGGI"].groupby('Regional').size().reset_index(name='Toko'), 
                             x='Regional', y='Toko', title="Jumlah Toko Berisiko per Wilayah")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_right:
            fig_scat = px.scatter(agg, x="Recency", y="Tonnage_Drop", color="Level_Risiko", 
                                  size="Monetary", title="Peta Risiko: Recency vs Tonnage Drop")
            st.plotly_chart(fig_scat, use_container_width=True)

else:
    st.info("👋 Selamat datang! Silakan unggah data transaksi gabungan 2025 & 2026 di panel kiri untuk melihat prediksi risiko churn.")
