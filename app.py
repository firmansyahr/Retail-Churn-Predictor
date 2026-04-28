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
st.set_page_config(page_title="Retail Churn Intelligence v5.1", page_icon="🎯", layout="wide")

MODEL_DIR = "models/"
REGIONAL_MAP = {'REGIONAL 1': 'SP', 'REGIONAL 2': 'SMBR', 'REGIONAL 6': 'ST'}

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #0052cc; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    h1, h2, h3 {color: #1e293b;}
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 2. MASTER DATA PIPELINE (DIOPTIMALISASI UNTUK HEMAT RAM)
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def master_data_pipeline(df, df_loyalty):
    """Satu fungsi raksasa ini memproses semuanya lalu membuang data mentah dari RAM"""
    
    # --- A. CLEANSING ---
    for col in ['Tanggal Transaksi', 'Created_at_Dist']:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in ['Harga', 'Zak Quantity', 'TON Quantity', 'Weight_Est', 'Harga_Per_KG']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'ID Toko' in df.columns:
        df['ID Toko'] = pd.to_numeric(df['ID Toko'], errors='coerce')
        df.loc[df['ID Toko'] == 0, 'ID Toko'] = np.nan

    text_cols = ['Nama Toko', 'Kabupaten Toko', 'Provinsi Toko', 'Area AP Toko', 
                 'Brands', 'Cluster Pareto', 'Tipe Customer', 'SSM', 'ASM', 'TSO']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().replace({'NAN':np.nan, 'NONE':np.nan, '':np.nan})
            
    if 'Area AP Toko' in df.columns: df['Area AP Toko'] = df['Area AP Toko'].map(REGIONAL_MAP).fillna(df['Area AP Toko'])
    
    # Pengaman Bug Teks (Bulletproof Lookup Key)
    if 'Kabupaten Toko' in df.columns:
        df['Lookup_Key'] = df['Nama Toko'].astype(str).fillna('') + '_' + df['Kabupaten Toko'].astype(str).fillna('')
    else:
        df['Lookup_Key'] = df['Nama Toko'].astype(str).fillna('')

    # --- B. SMART BACKFILL ---
    def _fill(d, col, key='Lookup_Key'):
        if col not in d.columns: return d
        m = d.dropna(subset=[col]).drop_duplicates(key).set_index(key)[col]
        d[col] = d[col].fillna(d[key].map(m))
        return d

    df = _fill(df, 'ID Toko')
    if 'ID Toko' in df.columns and df['ID Toko'].isnull().any():
        max_id = int(df['ID Toko'].max(skipna=True)) if not df['ID Toko'].isnull().all() else 1000
        nk = df[df['ID Toko'].isnull()]['Lookup_Key'].unique()
        new_ids = {k: (max_id*10)+i for i,k in enumerate(nk, 1)}
        df['ID Toko'] = df['ID Toko'].fillna(df['Lookup_Key'].map(new_ids))

    df = _fill(df, 'Cluster Pareto')
    if 'Cluster Pareto' in df.columns: df['Cluster Pareto'] = df['Cluster Pareto'].fillna('BRONZE')

    if 'Area AP Toko' in df.columns and 'Provinsi Toko' in df.columns:
        df = _fill(df, 'Area AP Toko')
        pv = df.dropna(subset=['Provinsi Toko','Area AP Toko']).groupby('Provinsi Toko')['Area AP Toko'].agg(lambda x: x.mode()[0] if len(x) else np.nan)
        df['Area AP Toko'] = df['Area AP Toko'].fillna(df['Provinsi Toko'].map(pv)).fillna('UNKNOWN')

    for col in ['SSM','ASM','TSO', 'Tipe Customer', 'Brands']:
        if col in df.columns:
            df = _fill(df, col)
            df[col] = df[col].fillna('UNKNOWN')
            
    if 'TON Quantity' in df.columns and 'Harga_Per_KG' in df.columns:
        df['Est_Spend'] = df['TON Quantity'] * 1000 * df['Harga_Per_KG'].fillna(0)

    # --- C. FEATURE ENGINEERING ---
    cutoff = df['Tanggal Transaksi'].max()
    RECENT_DAYS, PAST_DAYS = 30, 120

    df_tr = df[df['Tanggal Transaksi'] <= cutoff].copy()
    if df_tr.empty: return pd.DataFrame(), cutoff
    
    df_tr['Days_Ago'] = (cutoff - df_tr['Tanggal Transaksi']).dt.days

    agg_kwargs = {
        'last_trx': ('Tanggal Transaksi', 'max'),
        'Frequency': ('No Transaksi', 'count'),
        'Total_Ton': ('TON Quantity', 'sum'),
        'Monetary': ('Est_Spend', 'sum'),
        'Avg_Harga_Per_KG': ('Harga_Per_KG', 'mean'),
        'Std_Harga_Per_KG': ('Harga_Per_KG', 'std'),
        'Regional': ('Area AP Toko', 'first'),
        'Cluster_Pareto': ('Cluster Pareto', 'first'),
        'Tipe_Customer': ('Tipe Customer', 'first'),
        'Dominant_Brand': ('Brands', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        'Num_Brands': ('Brands', 'nunique'),
        'Num_Kab': ('Kabupaten Toko', 'nunique') if 'Kabupaten Toko' in df_tr.columns else ('ID Toko', 'count'),
        'Num_Produk': ('Kode Produk', 'nunique') if 'Kode Produk' in df_tr.columns else ('ID Toko', 'count'),
        'Dominant_SSM': ('SSM', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        'Dominant_ASM': ('ASM', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
        'Dominant_TSO': ('TSO', lambda x: x.mode()[0] if len(x) else 'UNKNOWN'),
    }
    agg = df_tr.groupby('ID Toko').agg(**agg_kwargs)
    
    agg['Recency'] = (cutoff - agg['last_trx']).dt.days
    agg['Last_Trx_Month'] = agg['last_trx'].dt.month

    last_price = df_tr.sort_values('Tanggal Transaksi').groupby('ID Toko')['Harga_Per_KG'].last()
    agg['Price_Delta'] = last_price - agg['Avg_Harga_Per_KG']

    recent = df_tr[df_tr['Days_Ago'] <= RECENT_DAYS].groupby('ID Toko')['TON Quantity'].sum()
    past = (df_tr[(df_tr['Days_Ago'] > RECENT_DAYS) & (df_tr['Days_Ago'] <= PAST_DAYS)].groupby('ID Toko')['TON Quantity'].sum() / 3)
    agg['Tonnage_Drop'] = (recent / (past + 0.001)).fillna(0).clip(upper=10)

    def avg_gap_fn(x):
        s = x.sort_values()
        return s.diff().dt.days.mean() if len(s) >= 2 else np.nan
    agg['Avg_Gap'] = df_tr.groupby('ID Toko')['Tanggal Transaksi'].apply(avg_gap_fn).fillna(agg['Recency'])

    if 'Created_at_Dist' in df_tr.columns:
        lag = df_tr.groupby('ID Toko').apply(lambda x: (x['Tanggal Transaksi'] - x['Created_at_Dist']).dt.days.mean())
        agg['Avg_Input_Lag'] = lag.fillna(0).clip(lower=0, upper=30)

    reg_stats = df_tr.groupby('Area AP Toko').agg(Reg_Avg_Ton=('TON Quantity', 'mean'), Reg_Avg_Harga=('Harga_Per_KG', 'mean'))
    agg = agg.join(reg_stats, on='Regional')
    agg['Ton_vs_Regional']   = agg['Total_Ton'] / (agg['Reg_Avg_Ton'] + 0.001)
    agg['Harga_vs_Regional'] = agg['Avg_Harga_Per_KG'] / (agg['Reg_Avg_Harga'] + 0.001)

    if len(agg['Recency'].dropna()) > 0:
        r_labels = pd.qcut(agg['Recency'], q=5, duplicates='drop', labels=False)
        agg['R_Score'] = (r_labels.max() - r_labels + 1).astype(float)
    else: agg['R_Score'] = 1.0

    for col in ['Frequency', 'Monetary']:
        if len(agg[col].dropna()) > 0:
            agg[col[0]+'_Score'] = (pd.qcut(agg[col], q=5, duplicates='drop', labels=False) + 1).astype(float)
        else: agg[col[0]+'_Score'] = 1.0

    agg['RFM_Score'] = agg[['R_Score', 'F_Score', 'M_Score']].mean(axis=1).round(2)

    def rfm_seg(row):
        r,f,m = row['R_Score'], row['F_Score'], row['M_Score']
        if r>=4 and f>=4 and m>=4:   return 'Champions'
        elif r>=3 and f>=3:          return 'Loyal'
        elif r>=4 and f<=2:          return 'New'
        elif r>=3 and f>=2 and m>=3: return 'Potential'
        elif r<=2 and f>=3:          return 'At Risk'
        elif r<=2 and f<=2:          return 'Lost'
        else:                        return 'Needs Attention'
    agg['RFM_Segment'] = agg.apply(rfm_seg, axis=1)

    c_map = {'BRONZE':1, 'SILVER':2, 'GOLD':3, 'PLATINUM':4, 'SUPER PLATINUM':5}
    agg['Cluster_Score'] = agg['Cluster_Pareto'].map(c_map).fillna(1).astype(int)

    if df_loyalty is not None and not df_loyalty.empty:
        col_id = df_loyalty.columns[0]
        lid = set(df_loyalty[col_id].astype(str).str.strip())
        agg['Is_Loyalty'] = np.where(agg.index.astype(str).isin(lid), 1, 0)
    else:
        agg['Is_Loyalty'] = 0

    return agg, cutoff

def encode_features(agg, features_list):
    if agg.empty: return pd.DataFrame()
    enc_cols = ['Regional', 'Dominant_Brand', 'Tipe_Customer', 'Dominant_SSM', 'Dominant_ASM', 'Dominant_TSO']
    enc_cols = [c for c in enc_cols if c in agg.columns]
    
    enc = pd.get_dummies(agg, columns=enc_cols, prefix=enc_cols, dtype=int)
    enc.columns = [re.sub(r'[\[\]<>{}:",\s]', '_', str(c)) for c in enc.columns]
    
    enc = enc.reindex(columns=features_list, fill_value=0)
    enc = enc.fillna(0)
    return enc[features_list]

# ════════════════════════════════════════════════════════════
# 3. SIDEBAR & DATA LOADING
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2601/2601112.png", width=60)
    st.title("📂 Input Data")
    st.info("💡 Unggah file transaksi gabungan (Min. 6 Bulan Terakhir) untuk akurasi terbaik.")
    uploaded_data = st.file_uploader("1. Transaksi (Parquet/CSV)", type=['parquet', 'csv'])
    uploaded_loyalty = st.file_uploader("2. Data Loyalty (CSV)", type=['csv'])

try:
    model = joblib.load(os.path.join(MODEL_DIR, "model_best.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f: meta = json.load(f)
    features_list = meta['features']
    threshold = float(meta.get('thresholds', {}).get('best_model', 0.26))
except Exception as e:
    st.error(f"🚨 Gagal memuat model. Error: {e}")
    st.stop()

# ════════════════════════════════════════════════════════════
# 4. DASHBOARD EXECUTION & VISUALIZATIONS
# ════════════════════════════════════════════════════════════

if uploaded_data:
    uploaded_data.seek(0)
    df_raw = pd.read_parquet(uploaded_data) if uploaded_data.name.endswith('.parquet') else pd.read_csv(uploaded_data)
    
    df_l = None
    if uploaded_loyalty:
        uploaded_loyalty.seek(0)
        df_l = pd.read_csv(uploaded_loyalty)

    st.title("🎯 Retail Churn Intelligence")
    st.caption(f"🤖 Powered by AI Random Forest (Threshold: {threshold*100:.1f}%)")
    
    with st.spinner("Mengolah ratusan parameter dan segmentasi perilaku toko..."):
        # Hanya "agg" yang keluar dari pipeline, memory mentah aman!
        agg, current_cutoff = master_data_pipeline(df_raw, df_l)
        
        # Hapus df_raw dari memory saat ini juga
        del df_raw
        
        X_pred = encode_features(agg, features_list)
        probs = model.predict_proba(X_pred)[:, 1]
        agg['Prob_Churn'] = probs
        agg['Level_Risiko'] = np.where(probs > threshold, "🚨 TINGGI", "✅ AMAN")
        
        t_risk = len(agg[agg['Level_Risiko'] == "🚨 TINGGI"])
        rev_risk = agg[agg['Level_Risiko'] == "🚨 TINGGI"]['Monetary'].sum()

    # --- TOP LEVEL METRICS ---
    st.write(f"📅 **Data Terbaru s/d:** `{current_cutoff.strftime('%d %B %Y')}`")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Toko Aktif", f"{len(agg):,}")
    m2.metric("Toko Prediksi Churn", f"{t_risk:,}", f"{(t_risk/len(agg)*100):.1f}% Churn Rate", delta_color="inverse")
    m3.metric("Toko VIP Berisiko (Champions/Loyal)", len(agg[(agg['Level_Risiko'] == "🚨 TINGGI") & (agg['RFM_Segment'].isin(['Champions','Loyal']))]))
    m4.metric("Potensi Rupiah Hilang", f"Rp {rev_risk/1e9:.2f} Miliar" if rev_risk > 1e9 else f"Rp {rev_risk/1e6:.1f} Juta", delta_color="inverse")

    st.markdown("---")
    
    # --- TABS VISUALISASI ---
    tab1, tab2, tab3 = st.tabs(["📊 Churn Analytics & Watchlist", "👥 Segmentasi Pelanggan (RFM)", "📈 Performa Wilayah & Pareto"])
    
    with tab1:
        st.subheader("📋 Toko Prioritas Penyelamatan (Top Revenue at Risk)")
        top_rescue = agg[agg['Level_Risiko'] == "🚨 TINGGI"].sort_values('Monetary', ascending=False).head(5)
        if not top_rescue.empty:
            cols = st.columns(5)
            for i, (idx, row) in enumerate(top_rescue.iterrows()):
                if i < 5:
                    with cols[i]:
                        st.info(f"**Toko ID:** {idx}\n\n**Segmen:** {row['RFM_Segment']}\n\n**Potensi:** Rp {row['Monetary']/1e6:.1f} Jt\n\n**Prob:** {row['Prob_Churn']*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🗂️ Tabel Eksekusi Watchlist")
        regs = st.multiselect("Filter Regional", options=agg['Regional'].unique(), default=agg['Regional'].unique())
        df_view = agg[agg['Regional'].isin(regs)].sort_values('Prob_Churn', ascending=False)
        
        display_cols = ['Regional', 'Cluster_Pareto', 'RFM_Segment', 'Recency', 'Tonnage_Drop', 'Monetary', 'Prob_Churn', 'Level_Risiko']
        df_display = df_view[display_cols].copy()
        df_display['Prob_Churn'] = (df_display['Prob_Churn'] * 100).round(1).astype(str) + "%"
        df_display['Monetary'] = "Rp " + (df_display['Monetary'] / 1e6).round(1).astype(str) + " Jt"
        df_display['Tonnage_Drop'] = df_display['Tonnage_Drop'].round(2)
        
        st.dataframe(df_display, use_container_width=True, height=400)
        st.download_button("📥 Ekspor Watchlist (.csv)", df_view.to_csv().encode('utf-8'), "watchlist_churn.csv")

    with tab2:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("Distribusi Segmen Pelanggan")
            rfm_counts = agg['RFM_Segment'].value_counts().reset_index()
            rfm_counts.columns = ['Segmen', 'Jumlah Toko']
            fig_rfm = px.pie(rfm_counts, values='Jumlah Toko', names='Segmen', hole=0.4,
                             color='Segmen', color_discrete_map={
                                 'Champions': '#10b981', 'Loyal': '#3b82f6', 'Potential': '#0ea5e9',
                                 'New': '#8b5cf6', 'Needs Attention': '#f59e0b', 'At Risk': '#f97316', 'Lost': '#ef4444'
                             })
            st.plotly_chart(fig_rfm, use_container_width=True)
            
        with col_right:
            st.subheader("Churn Rate Berdasarkan Segmen")
            churn_by_seg = agg.groupby('RFM_Segment').apply(lambda x: (x['Level_Risiko']=='🚨 TINGGI').mean() * 100).reset_index(name='Churn Rate (%)')
            fig_bar_seg = px.bar(churn_by_seg, x='RFM_Segment', y='Churn Rate (%)', text_auto='.1f',
                                 color='RFM_Segment', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_bar_seg, use_container_width=True)

    with tab3:
        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Tingkat Churn per Regional")
            churn_reg = agg.groupby('Regional').apply(lambda x: (x['Level_Risiko']=='🚨 TINGGI').mean() * 100).reset_index(name='Churn Rate (%)')
            fig_reg = px.bar(churn_reg, x='Regional', y='Churn Rate (%)', text_auto='.1f', 
                             color='Regional', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_reg, use_container_width=True)
            
        with col_d:
            st.subheader("Tingkat Churn per Cluster Pareto")
            churn_par = agg.groupby('Cluster_Pareto').apply(lambda x: (x['Level_Risiko']=='🚨 TINGGI').mean() * 100).reset_index(name='Churn Rate (%)')
            order_cluster = ['SUPER PLATINUM', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE']
            
            # Pengaman untuk membuang segmen UNKNOWN agar plotly tidak error
            churn_par = churn_par[churn_par['Cluster_Pareto'].isin(order_cluster)]
            churn_par['Cluster_Pareto'] = pd.Categorical(churn_par['Cluster_Pareto'], categories=order_cluster, ordered=True)
            churn_par = churn_par.sort_values('Cluster_Pareto')
            
            fig_par = px.bar(churn_par, x='Cluster_Pareto', y='Churn Rate (%)', text_auto='.1f',
                             color='Cluster_Pareto', color_discrete_sequence=px.colors.sequential.Agal_r)
            st.plotly_chart(fig_par, use_container_width=True)

else:
    st.markdown("<h2 style='text-align: center; color: #888; margin-top: 50px;'>Selamat Datang di Retail Churn Executive Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Silakan unggah data transaksi pada panel di sebelah kiri untuk mengaktifkan AI dan Visualisasi.</p>", unsafe_allow_html=True)
