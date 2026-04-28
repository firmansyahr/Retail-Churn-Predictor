import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import plotly.express as px

# ════════════════════════════════════════════════════════════
# 1. KONFIGURASI GLOBAL
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="Retail Churn Intelligence", page_icon="🎯", layout="wide")

MODEL_DIR = "models/"
REGIONAL_MAP = {'REGIONAL 1': 'SP', 'REGIONAL 2': 'SMBR', 'REGIONAL 6': 'ST'}

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stMetric {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #0052cc;}
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS (HANYA INFERENCE / PREDIKSI)
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def cleansing_data(df):
    df = df.copy()
    for col in ['Tanggal Transaksi', 'Created_at_Dist']:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in ['Harga', 'TON Quantity', 'Harga_Per_KG']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    text_cols = ['Nama Toko', 'Area AP Toko', 'Brands', 'Cluster Pareto']
    for col in text_cols:
        if col in df.columns: df[col] = df[col].astype(str).str.strip().str.upper()
            
    if 'Area AP Toko' in df.columns: df['Area AP Toko'] = df['Area AP Toko'].map(REGIONAL_MAP).fillna(df['Area AP Toko'])
    df['Est_Spend'] = df['TON Quantity'] * df['Harga_Per_KG'].fillna(0)
    return df

@st.cache_data(show_spinner=False)
def feature_engineering(df, df_loyalty, cutoff):
    df_tr = df[df['Tanggal Transaksi'] <= cutoff].copy()
    if df_tr.empty: return pd.DataFrame()

    # Menggunakan metode asli dari Colab agar sinkron dengan model lama
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

def encode_features(agg, features_list):
    if agg.empty: return pd.DataFrame()
    enc = pd.get_dummies(agg, columns=['Regional'], prefix='Regional', dtype=int)
    enc.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in enc.columns]
    
    # Menyesuaikan kolom dengan metadata model secara paksa
    enc = enc.reindex(columns=features_list, fill_value=0)
    return enc

# ════════════════════════════════════════════════════════════
# 3. KONTROL PANEL & LOAD DATA
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2601/2601112.png", width=80)
    st.title("Data Input")
    st.info("💡 Unggah gabungan data transaksi historis (Misal: 10 bulan terakhir) agar prediksi akurat.")
    uploaded_data = st.file_uploader("1. Transaksi (CSV/Parquet)", type=['parquet', 'csv'])
    uploaded_loyalty = st.file_uploader("2. Data Loyalty (Opsional)", type=['csv'])
    
    df_raw, df_l = None, None
    if uploaded_data:
        uploaded_data.seek(0)
        df_raw = pd.read_parquet(uploaded_data) if uploaded_data.name.endswith('.parquet') else pd.read_csv(uploaded_data)
    if uploaded_loyalty:
        uploaded_loyalty.seek(0)
        df_l = pd.read_csv(uploaded_loyalty)

# ════════════════════════════════════════════════════════════
# 4. DASHBOARD & PREDIKSI EWS
# ════════════════════════════════════════════════════════════
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model_best.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
    features_list = meta['features']
except:
    st.error("🚨 Model tidak ditemukan! Pastikan 'model_best.pkl' dan 'metadata.json' dari Colab ada di folder 'models/'.")
    st.stop()

if df_raw is not None:
    st.title("🎯 Early Warning System - Churn Retail")
    
    with st.spinner("Menghitung risiko toko..."):
        df_clean = cleansing_data(df_raw)
        current_cutoff = df_clean['Tanggal Transaksi'].max()
        
        agg = feature_engineering(df_clean, df_l, current_cutoff)
        X_pred = encode_features(agg, features_list)
        
        probs = model.predict_proba(X_pred)[:, 1]
        agg['Prob_Churn'] = probs
        agg['Level_Risiko'] = np.where(probs > 0.26, "🚨 TINGGI", "✅ AMAN")
        
        total_toko = len(agg)
        toko_risiko = len(agg[agg['Level_Risiko'] == "🚨 TINGGI"])
        churn_rate = (toko_risiko / total_toko) * 100 if total_toko > 0 else 0
        revenue_at_risk = agg[agg['Level_Risiko'] == "🚨 TINGGI"]['Monetary'].sum()

    st.markdown(f"**Prediksi dihitung berdasarkan data s/d:** `{current_cutoff.strftime('%d %B %Y')}`")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Toko Terpantau", f"{total_toko:,}")
    c2.metric("Toko Berisiko (Prediksi)", f"{toko_risiko:,}", f"{churn_rate:.1f}% Churn Rate", delta_color="inverse")
    c3.metric("Loyalty Risk", len(agg[(agg['Level_Risiko'] == "🚨 TINGGI") & (agg['Is_Loyalty'] == 1)]))
    val_str = f"Rp {revenue_at_risk/1e9:.2f} Miliar" if revenue_at_risk > 1e9 else f"Rp {revenue_at_risk/1e6:.2f} Juta"
    c4.metric("Potensi Rupiah Hilang", val_str, "Revenue at Risk", delta_color="inverse")

    st.markdown("---")
    
    # TABEL WATCHLIST
    st.subheader("📋 Daftar Toko Prioritas Intervensi (Watchlist)")
    f_col1, f_col2 = st.columns(2)
    region_filter = f_col1.multiselect("Pilih Regional", options=agg['Regional'].unique(), default=agg['Regional'].unique())
    loyalty_filter = f_col2.selectbox("Status Loyalty", options=["Semua", "Hanya Peserta Loyalty", "Non-Loyalty"])
    
    display_df = agg[agg['Regional'].isin(region_filter)].copy()
    if loyalty_filter == "Hanya Peserta Loyalty": display_df = display_df[display_df['Is_Loyalty'] == 1]
    elif loyalty_filter == "Non-Loyalty": display_df = display_df[display_df['Is_Loyalty'] == 0]
        
    display_df = display_df.sort_values('Prob_Churn', ascending=False)
    tabel_tayang = display_df[['Regional', 'Cluster_Pareto', 'Recency', 'Tonnage_Drop', 'Prob_Churn', 'Level_Risiko']].copy()
    tabel_tayang['Prob_Churn'] = (tabel_tayang['Prob_Churn'] * 100).round(1).astype(str) + "%"
    tabel_tayang['Tonnage_Drop'] = tabel_tayang['Tonnage_Drop'].round(2)
    
    st.dataframe(tabel_tayang, use_container_width=True, height=400)
    st.download_button("📥 Ekspor Watchlist ke CSV", display_df.to_csv().encode('utf-8'), "watchlist_eksekusi.csv")

else:
    st.markdown("<h2 style='text-align: center; color: #888; margin-top: 50px;'>Selamat Datang di Retail Churn Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Silakan unggah data transaksi (CSV/Parquet) untuk melihat prediksi terbaru.</p>", unsafe_allow_html=True)
