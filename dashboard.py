import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. SAYFA YAPILANDIRMASI VE TEMA
# ==========================================
st.set_page_config(
    page_title="Sosyal Medya BI & ML Analiz Paneli",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS Tasarımı
st.markdown("""
    <style>
    .main { padding: 1.5rem; }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_dict=True)


# ==========================================
# 2. VERİ YÜKLEME VE ÖN İŞLEME
# ==========================================
@st.cache_data
def load_data():
    # İşlenmiş zaman serisi verisini yükle
    df = pd.read_csv("zaman_entegreli_veri.csv")
    
    # Tarih sütununu datetime formatına çevir
    if 'posted_datetime' in df.columns:
        df['posted_datetime'] = pd.to_datetime(df['posted_datetime'])
        df = df.sort_values('posted_datetime')
    
    # Ghost Day (Sıfır etkileşimli/eksik veri) ve NaN Temizliği
    df_clean = df.dropna().copy()
    if 'engagement_score' in df_clean.columns:
        df_clean = df_clean[df_clean['engagement_score'] > 0]
        
    return df_clean

try:
    df = load_data()
except Exception as e:
    st.error(f"Veri dosyası yüklenirken bir hata oluştu: {e}")
    st.stop()


# ==========================================
# 3. YAN MENÜ (SIDEBAR) & FİLTRELER
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/dashboard.png", width=80)
st.sidebar.title("📌 Kontrol Paneli")
st.sidebar.markdown("---")

# Tarih Aralığı Filtresi
if 'posted_datetime' in df.columns:
    min_date = df['posted_datetime'].min().date()
    max_date = df['posted_datetime'].max().date()
    
    selected_date_range = st.sidebar.date_input(
        "📅 Tarih Aralığı Seçin",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        mask = (df['posted_datetime'].dt.date >= start_date) & (df['posted_datetime'].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df.copy()
else:
    filtered_df = df.copy()

st.sidebar.markdown("---")
st.sidebar.info(f"🟢 **Aktif Veri Sayısı:** {len(filtered_df)} Gün")


# ==========================================
# 4. ANA BAŞLIK VE ÖZET METRİKLER (KPIs)
# ==========================================
st.title("📊 Sosyal Medya Performans & Tahmin Paneli")
st.caption("Zaman Serisi Entegreli Etkileşim ve Duygu Analizi Dashboard'u")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_engagement = filtered_df['engagement_score'].mean()
    st.metric("🎯 Ort. Etkileşim Skoru", f"{avg_engagement:.2f}", delta=f"%{avg_engagement*100:.1f} Başarı")

with col2:
    total_likes = filtered_df['like_count'].sum() if 'like_count' in filtered_df.columns else 0
    st.metric("❤️ Toplam Beğeni", f"{total_likes:,.0f}")

with col3:
    total_posts = filtered_df['gunluk_post_sayisi'].sum() if 'gunluk_post_sayisi' in filtered_df.columns else len(filtered_df)
    st.metric("📝 Toplam İçerik", f"{total_posts:,.0f}")

with col4:
    avg_sentiment = filtered_df['sentiment_score'].mean() if 'sentiment_score' in filtered_df.columns else 0
    sentiment_label = "Pozitif" if avg_sentiment > 0.05 else ("Negatif" if avg_sentiment < -0.05 else "Nötr")
    st.metric("🎭 Ort. Duygu Skoru", f"{avg_sentiment:.2f}", delta=sentiment_label)

st.markdown("---")


# ==========================================
# 5. GRAFİK VE ANALİZ SEKMELERİ
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Zaman Serisi Trendleri", "🔍 Değişken Korelasyonları", "🎛️ Etkileşim Simülatörü"])

# --- TAB 1: ZAMAN SERİSİ TRENDLERİ (Sadece Streamlit Chart) ---
with tab1:
    st.subheader("📆 Zamana Bağlı Etkileşim Değişimi")
    
    if 'posted_datetime' in filtered_df.columns:
        # Zaman serisi verisini indeks haline getirip çizdiriyoruz
        chart_data = filtered_df.set_index('posted_datetime')
        cols_to_plot = [c for c in ['engagement_score', 'Rolling_Mean_7'] if c in chart_data.columns]
        
        st.line_chart(chart_data[cols_to_plot], height=400)
    else:
        st.warning("Zaman serisi grafiği için 'posted_datetime' sütunu bulunamadı.")

# --- TAB 2: KORELASYON VE METRİKLER (Sadece Streamlit Chart) ---
with tab2:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("💬 Duygu Skoru vs Etkileşim Skoru")
        if 'sentiment_score' in filtered_df.columns and 'engagement_score' in filtered_df.columns:
            st.scatter_chart(
                filtered_df,
                x='sentiment_score',
                y='engagement_score',
                height=350
            )
            
    with col_right:
        st.subheader("📋 Veri Özeti (Temizlenmiş)")
        display_cols = [c for c in ['posted_datetime', 'engagement_score', 'sentiment_score', 'gunluk_post_sayisi', 'like_count'] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].head(10), use_container_width=True)

# --- TAB 3: ETKİLEŞİM TAHMİN SİMÜLATÖRÜ ---
with tab3:
    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.subheader("🎛️ Etkileşim Tahmin Simülatörü")
        st.caption("Geçmiş günlerin başarı skorlarına (0.0 - 1.0) göre yarınki skoru simüle edin:")
        
        # Girdiler 0.0 - 1.0 skalasında
        input_lag1 = st.slider("Dünkü Etkileşim Skoru (Lag_1)", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        input_lag2 = st.slider("Önceki Günkü Skoru (Lag_2)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
        input_rolling = st.slider("3 Günlük Ortalama Skor (Rolling_3)", min_value=0.0, max_value=1.0, value=0.52, step=0.05)
        input_sentiment = st.slider("Duygu Skoru (Sentiment)", min_value=-1.0, max_value=1.0, value=0.10, step=0.1)

        # 0-1 arası skor hesabı
        estimated_score = (input_rolling * 0.45) + (input_lag1 * 0.30) + (input_lag2 * 0.20) + (input_sentiment * 0.05)
        estimated_score = max(0.0, min(1.0, estimated_score))

        st.markdown("<br>", unsafe_allow_dict=True)
        predict_btn = st.button("🚀 Etkileşimi Tahmin Et", type="primary", use_container_width=True)

    with sim_col2:
        st.subheader("🎯 Tahmin Sonucu")
        if predict_btn:
            st.success(f"### Tahmini Etkileşim Skoru: `{estimated_score:.2f}`")
            st.metric("Tahmini Başarı Oranı", f"%{estimated_score*100:.1f}")
            
            if estimated_score >= 0.70:
                st.balloons()
                st.info("🔥 **Yüksek Performans:** Gönderilerin keşfete düşme ve viral olma olasılığı çok yüksek!")
            elif estimated_score >= 0.40:
                st.info("📊 **Ortalama Performans:** Standart ve kararlı bir etkileşim seviyesi bekleniyor.")
            else:
                st.warning("📉 **Düşük Performans:** İçerik stratejisini ve zamanlamayı gözden geçirmekte fayda var.")
        else:
            st.info("👈 Sol taraftaki parametreleri ayarlayıp **'Etkileşimi Tahmin Et'** butonuna basın.")

# ==========================================
# 6. ALT BİLGİ (FOOTER)
# ==========================================
st.markdown("---")
st.caption("🤖 ML & BI Dashboard | Zaman Serisi & Duygu Analizi Entegrasyonu")
