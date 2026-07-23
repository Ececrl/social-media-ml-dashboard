import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sosyal Medya Analiz & ML Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# 2. VERİ YÜKLEME (Önbellek kullanarak hızlı çalışma)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Veri setinizin yolunu kontrol edin (örneğin 'data.csv' veya 'social_media.csv')
    df = pd.read_csv("dataset.csv") 
    return df

try:
    df = load_data()
except Exception as e:
    # Veri seti okunamazsa örnek veri oluşturma (Yedek)
    st.warning("Yerel veri seti bulunamadı, demo verisi yükleniyor...")
    data = {
        'Platform': np.random.choice(['Instagram', 'Twitter', 'LinkedIn', 'TikTok'], 500),
        'Post_Type': np.random.choice(['Image', 'Video', 'Text'], 500),
        'Followers': np.random.randint(1000, 100000, 500),
        'Likes': np.random.randint(50, 5000, 500),
        'Comments': np.random.randint(5, 500, 500),
        'Shares': np.random.randint(0, 200, 500)
    }
    df = pd.DataFrame(data)

# ---------------------------------------------------------
# 3. SIDEBAR (KENAR ÇUBUĞU) FİLTRELERİ
# ---------------------------------------------------------
st.sidebar.title("🔍 Veri Filtreleme")

# Platform Filtresi
if 'Platform' in df.columns:
    platforms = df['Platform'].unique().tolist()
    selected_platforms = st.sidebar.multiselect(
        "Platform Seçin:",
        options=platforms,
        default=platforms
    )
    df_filtered = df[df['Platform'].isin(selected_platforms)]
else:
    df_filtered = df.copy()

# İçerik Türü Filtresi
if 'Post_Type' in df.columns:
    post_types = df['Post_Type'].unique().tolist()
    selected_types = st.sidebar.multiselect(
        "İçerik Türü Seçin:",
        options=post_types,
        default=post_types
    )
    df_filtered = df_filtered[df_filtered['Post_Type'].isin(selected_types)]

# ---------------------------------------------------------
# 4. ANA BAŞLIK VE METRİK KARTLARI (KPI PANELİ)
# ---------------------------------------------------------
st.title("📱 Sosyal Medya Performance & ML Dashboard")
st.markdown("Filtrelenmiş verilere ait anlık performans metrikleri ve makine öğrenmesi tahmin modeli.")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📌 Toplam Gönderi",
        value=f"{len(df_filtered):,}"
    )

with col2:
    avg_likes = df_filtered['Likes'].mean() if 'Likes' in df_filtered.columns else 0
    st.metric(
        label="❤️ Ort. Beğeni",
        value=f"{avg_likes:.1f}"
    )

with col3:
    avg_comments = df_filtered['Comments'].mean() if 'Comments' in df_filtered.columns else 0
    st.metric(
        label="💬 Ort. Yorum",
        value=f"{avg_comments:.1f}"
    )

with col4:
    max_likes = df_filtered['Likes'].max() if 'Likes' in df_filtered.columns else 0
    st.metric(
        label="🔥 Maksimum Beğeni",
        value=f"{max_likes:,}"
    )

st.markdown("---")

# ---------------------------------------------------------
# 5. SEKMELİ DÜZEN (TABS)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analiz & Grafikler", "🤖 ML Beğeni Tahmini", "📋 Veri Tablosu"])

# TAB 1: GRAFİKLER VE GÖRSELLEŞTİRME
with tab1:
    st.subheader("Gönderi Performans Analizleri")
    
    g1, g2 = st.columns(2)
    
    with g1:
        st.markdown("**Platformlara Göre Ortalama Beğeni**")
        if 'Platform' in df_filtered.columns and 'Likes' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_filtered, x='Platform', y='Likes', ax=ax, palette='Blues_d')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Bu grafik için 'Platform' ve 'Likes' sütunları gereklidir.")
            
    with g2:
        st.markdown("**Beğeni ve Yorum Sayısı İlişkisi**")
        if 'Likes' in df_filtered.columns and 'Comments' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_filtered, x='Likes', y='Comments', hue='Platform' if 'Platform' in df_filtered.columns else None, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Bu grafik için 'Likes' ve 'Comments' sütunları gereklidir.")

# TAB 2: MAKİNE ÖĞRENMESİ (MODEL TAHMİNİ)
with tab2:
    st.subheader("🤖 Beğeni (Likes) Tahmin Modeli")
    st.write("Takipçi ve yorum sayısına göre tahmini beğeni oranını hesaplayın.")
    
    # Model Eğitimi (Örnek: Random Forest Regressor)
    if {'Followers', 'Comments', 'Likes'}.issubset(df.columns):
        X = df[['Followers', 'Comments']]
        y = df['Likes']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        st.success(f"Model Başarıyla Eğitildi! Model Performansı ($R^2$ Skoru): **{r2:.2f}**")
        
        st.markdown("---")
        st.markdown("#### 🔮 Canlı Tahmin Yapın")
        
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            input_followers = st.number_input("Takipçi Sayısı:", min_value=0, value=10000, step=500)
        with input_col2:
            input_comments = st.number_input("Beklenen Yorum Sayısı:", min_value=0, value=50, step=5)
            
        if st.button("Beğeni Sayısını Tahmin Et"):
            prediction = model.predict([[input_followers, input_comments]])
            st.balloons()
            st.metric(label="Tahmini Beğeni Sayısı", value=f"{int(prediction[0]):,} Beğeni")
    else:
        st.error("ML modeli için veri setinde 'Followers', 'Comments' ve 'Likes' sütunlarının bulunması gerekmektedir.")

# TAB 3: VERİ TABLOSU VE İNDİRME
with tab3:
    st.subheader("Filtrelenmiş Veri Seti")
    st.dataframe(df_filtered)
    
    # CSV İndirme Butonu
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Filtrelenmiş Veriyi CSV Olarak İndir",
        data=csv,
        file_name='filtrelenmis_sosyal_medya_verisi.csv',
        mime='text/csv'
    )
