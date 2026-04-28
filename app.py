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
st.set_page_config(page_title="Retail Churn Intelligence v4.3", page_icon="🎯", layout="wide")

MODEL_DIR = "models/"
REGIONAL_MAP = {'REGIONAL 1': 'SP', 'REGIONAL 2': 'SMBR', 'REGIONAL 6': 'ST'}

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stMetric {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #0052cc;}
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 2. CORE FUNCTIONS (IDENTIK 1:1 DENGAN COLAB)
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cleansing_data(df):
    df = df.copy()
    for col in ['Tanggal Transaksi', 'Created_at_Dist']:
        if col in df.columns: 
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    for col in ['Harga', 'Zak Quantity', 'TON Quantity', 'Weight_Est', 'Harga_Per_KG']:
        if col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    text_cols = ['Nama Toko', 'Kabupaten Toko', 'Area AP Toko', 'Brands', 
                 'Cluster Pareto', 'Tipe Customer', 'SSM', 'ASM', 'TSO']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().replace({'NAN':np.nan, 'NONE':np.nan, '':np.nan})
            
    if 'Area AP Toko' in df.columns: 
        df['Area AP Toko'] = df['Area AP Toko'].map(REGIONAL_MAP).fillna(df['Area AP Toko'])
        
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

    # FULL AGREGASI (Persis seperti di Notebook)
    agg = df_tr.groupby('ID Toko').agg(
        last_trx         = ('Tanggal Transaksi', 'max'),
        Frequency        = ('No Transaksi', 'count'),
        Total_Ton        = ('TON Quantity', 'sum'),
        Monetary         = ('Est_Spend', 'sum'),
        Avg_Harga_Per_KG = ('Harga_Per_KG', 'mean'),
        Std_Harga_Per_KG = ('Harga_Per_KG', 'std'),
        Regional         = ('Area AP Toko', 'first'),
        Cluster_Pareto   = ('Cluster Pareto', 'first'),
        Tipe_Customer    = ('Tipe Customer', 'first'),
        Dominant_Brand   = ('Brands', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        Num_Brands       = ('Brands', 'nunique'),
        Num_Kab          = ('Kabupaten Toko', 'nunique'),
        Num_Produk       = ('Kode Produk', 'nunique'),
        Dominant_SSM     = ('SSM', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        Dominant_ASM     = ('ASM', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        Dominant_TSO     = ('TSO', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
    )
    
    agg['Recency'] = (cutoff - agg['last_trx']).dt.days
    agg['Last_Trx_Month'] = agg['last_trx'].dt.month

    # Harga Fluktuasi
    last_price = df_tr.sort_values('Tanggal Transaksi').groupby('ID Toko')['Harga_Per_KG'].last()
    agg['Price_Delta'] = last_price - agg['Avg_Harga_Per_KG']

    # Tonnage Drop Logic
    recent = df_tr[df_tr['Days_Ago'] <= RECENT_DAYS].groupby('ID Toko')['TON Quantity'].sum()
    past = (df_tr[(df_tr['Days_Ago'] > RECENT_DAYS) & (df_tr['Days_Ago'] <= PAST_DAYS)]
            .groupby('ID Toko')['TON Quantity'].sum() / 3)
    agg['Tonnage_Drop'] = (recent / (past + 0.001)).fillna(0).clip(upper=10)

    # Menghitung Average Gap Transaksi
    def avg_gap_fn(x):
        s = x.sort_values()
        return s.diff().dt.days.mean() if len(s) >= 2 else np.nan
    agg['Avg_Gap'] = df_tr.groupby('ID Toko')['Tanggal Transaksi'].apply(avg_gap_fn).fillna(agg['Recency'])

    if 'Created_at_Dist' in df_tr.columns:
        lag = df_tr.groupby('ID Toko').apply(lambda x: (x['Tanggal Transaksi'] - x['Created_at_Dist']).dt.days.mean())
        agg['Avg_Input_Lag'] = lag.fillna(0).clip(lower=0, upper=30)

    # Performa Toko vs Rata-rata Regional
    reg_stats = df_tr.groupby('Area AP Toko').agg(
        Reg_Avg_Ton   = ('TON Quantity', 'mean'),
        Reg_Avg_Harga = ('Harga_Per_KG', 'mean')
    )
    agg = agg.join(reg_stats, on='Regional')
    agg['Ton_vs_Regional']   = agg['Total_Ton'] / (agg['Reg_Avg_Ton'] + 0.001)
    agg['Harga_vs_Regional'] = agg['Avg_Harga_Per_KG'] / (agg['Reg_Avg_Harga'] + 0.001)

    # RFM Scoring
    if len(agg['Recency'].dropna()) > 0:
        r_labels = pd.qcut(agg['Recency'], q=5, duplicates='drop', labels=False)
        agg['R_Score'] = (r_labels.max() - r_labels + 1).astype(float)
    else: agg['R_Score'] = 1.0

    for col in ['Frequency', 'Monetary']:
        if len(agg[col].dropna()) > 0:
            agg[col[0]+'_Score'] = (pd.qcut(agg[col], q=5, duplicates='drop', labels=False) + 1).astype(float)
        else: agg[col[0]+'_Score'] = 1.0

    agg['RFM_Score'] = agg[['R_Score', 'F_Score', 'M_Score']].mean(axis=1).round(2)
    c_map = {'BRONZE':1, 'SILVER':2, 'GOLD':3, 'PLATINUM':4, 'SUPER PLATINUM':5}
    agg['Cluster_Score'] = agg['Cluster_Pareto'].map(c_map).fillna(1).astype(int)

    if df_loyalty is not None and not df_loyalty.empty:
        col_id = df_loyalty.columns[0]
        lid = set(df_loyalty[col_id].astype(str).str.strip())
        agg['Is_Loyalty'] = np.where(agg.index.astype(str).isin(lid), 1, 0)
    else:
        agg['Is_Loyalty'] = 0

    return agg

def encode_features(agg, features_list):
    if agg.empty: return pd.DataFrame()
    
    enc_cols = ['Regional', 'Dominant_Brand', 'Tipe_Customer', 'Dominant_SSM', 'Dominant_ASM', 'Dominant_TSO']
    enc_cols = [c for c in enc_cols if c in agg.columns]
    
    enc = pd.get_dummies(agg, columns=enc_cols, prefix=enc_cols, dtype=int)
    enc.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in enc.columns]
    
    # [KUNCI PERBAIKAN]: Menambah kolom yang tidak ada, lalu MEMBESIHKAN NaN menjadi 0 
    enc = enc.reindex(columns=features_list, fill_value=0)
    enc = enc.fillna(0) # Menghindari Error "NaN" pada model
    
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
    # Dinamis threshold dari metadata Colab jika ada, jika tidak default 0.26
    threshold = float(meta.get('thresholds', {}).get('best_model', 0.26))
except Exception as e:
    st.error(f"🚨 Gagal memuat model. Error: {e}")
    st.stop()

# ════════════════════════════════════════════════════════════
# 4. DASHBOARD EXECUTION
# ════════════════════════════════════════════════════════════

if uploaded_data:
    uploaded_data.seek(0)
    df_raw = pd.read_parquet(uploaded_data) if uploaded_data.name.endswith('.parquet') else pd.read_csv(uploaded_data)
    df_l = pd.read_csv(uploaded_loyalty) if uploaded_loyalty else None

    st.title("🎯 Retail Churn Intelligence")
    
    with st.spinner("Menganalisis ratusan parameter perilaku toko..."):
        df_clean = cleansing_data(df_raw)
        current_cutoff = df_clean['Tanggal Transaksi'].max()
        
        agg = feature_engineering(df_clean, df_l, current_cutoff)
        X_pred = encode_features(agg, features_list)
        
        probs = model.predict_proba(X_pred)[:, 1]
        agg['Prob_Churn'] = probs
        
        # Penentuan Level Risiko
        agg['Level_Risiko'] = np.where(probs > threshold, "🚨 TINGGI", "✅ AMAN")
        
        t_risk = len(agg[agg['Level_Risiko'] == "🚨 TINGGI"])
        rev_risk = agg[agg['Level_Risiko'] == "🚨 TINGGI"]['Monetary'].sum()

    st.write(f"📅 **Data Terbaru:** {current_cutoff.strftime('%d %B %Y')}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Toko Aktif", f"{len(agg):,}")
    m2.metric("Risiko Churn", f"{t_risk:,}", f"{(t_risk/len(agg)*100):.1f}%")
    m3.metric("Loyalty Risk", len(agg[(agg['Level_Risiko'] == "🚨 TINGGI") & (agg['Is_Loyalty'] == 1)]))
    m4.metric("Revenue at Risk", f"Rp {rev_risk/1e6:.1f} Juta")

    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📋 Watchlist Prioritas", "📊 Analisis Wilayah"])
    
    with tab1:
        st.subheader("Daftar Toko Berisiko (Segera Intervensi)")
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
