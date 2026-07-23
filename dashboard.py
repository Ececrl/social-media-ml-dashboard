import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sosyal Medya Analiz & ML Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# 2. VERİ YÜKLEME
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # NOT: GitHub repondaki CSV dosyasının adını tam olarak buraya yaz (örn: social_media.csv)
    df = pd.read_csv("dataset.csv") 
    return df

try:
    df = load_data()
except Exception as e:
    # CSV bulunamazsa çalışacak Demo Veri
    st.warning("⚠️ Gerçek veri seti ('dataset.csv') bulunamadı, demo verisi ile gösteriliyor.")
    np.random.seed(42)
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

if 'Platform' in df.columns:
    platforms = df['Platform'].unique().tolist()
    selected_platforms = st.sidebar.multiselect("Platform Seçin:", options=platforms, default=platforms)
    df_filtered = df[df['Platform'].isin(selected_platforms)]
else:
    df_filtered = df.copy()

if 'Post_Type' in df.columns:
    post_types = df['Post_Type'].unique().tolist()
    selected_types = st.sidebar.multiselect("İçerik Türü Seçin:", options=post_types, default=post_types)
    df_filtered = df_filtered[df_filtered['Post_Type'].isin(selected_types)]

# ---------------------------------------------------------
# 4. KPI METRİK KARTLARI
# ---------------------------------------------------------
st.title("📱 Sosyal Medya Performance & ML Dashboard")
st.markdown("Filtrelenmiş verilere ait anlık performans metrikleri ve makine öğrenmesi karşılaştırma paneli.")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="📌 Toplam Gönderi", value=f"{len(df_filtered):,}")
with col2:
    avg_likes = df_filtered['Likes'].mean() if 'Likes' in df_filtered.columns else 0
    st.metric(label="❤️ Ort. Beğeni", value=f"{avg_likes:.1f}")
with col3:
    avg_comments = df_filtered['Comments'].mean() if 'Comments' in df_filtered.columns else 0
    st.metric(label="💬 Ort. Yorum", value=f"{avg_comments:.1f}")
with col4:
    max_likes = df_filtered['Likes'].max() if 'Likes' in df_filtered.columns else 0
    st.metric(label="🔥 Maksimum Beğeni", value=f"{max_likes:,}")

st.markdown("---")

# ---------------------------------------------------------
# 5. SEKMELER
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analiz & Grafikler", "🤖 Regresyon Modelleri Karşılaştırması", "📋 Veri Tablosu"])

# TAB 1: GRAFİKLER
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
    with g2:
        st.markdown("**Beğeni ve Yorum Sayısı İlişkisi**")
        if 'Likes' in df_filtered.columns and 'Comments' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_filtered, x='Likes', y='Comments', hue='Platform' if 'Platform' in df_filtered.columns else None, ax=ax)
            st.pyplot(fig)

# TAB 2: REGRESYON MODELLERİNİ KARŞILAŞTIRMA
with tab2:
    st.subheader("⚖️ Regresyon Modellerinin Performans Karşılaştırması")
    
    if {'Followers', 'Comments', 'Likes'}.issubset(df.columns):
        X = df[['Followers', 'Comments']]
        y = df['Likes']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modeller
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = []
        trained_models = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            results.append({"Model": name, "R² Skoru": round(r2, 4), "MAE": round(mae, 2), "RMSE": round(rmse, 2)})
            trained_models[name] = model
            
        results_df = pd.DataFrame(results)
        
        # Karşılaştırma Tablosu ve Grafiği
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            st.markdown("#### 📋 Metrik Karşılaştırma Tablosu")
            st.dataframe(results_df, use_container_width=True)
            
        with col_chart:
            st.markdown("#### 📊 R² Skoru Kıyaslaması")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=results_df, x='Model', y='R² Skoru', ax=ax, palette='viridis')
            plt.xticks(rotation=20)
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Canlı Tahmin Bölümü
        st.markdown("#### 🔮 Seçilen Model ile Canlı Beğeni Tahmini")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_model_name = st.selectbox("Tahmin Yapılacak Modeli Seçin:", list(models.keys()))
        with c2:
            input_followers = st.number_input("Takipçi Sayısı:", min_value=0, value=10000, step=500)
        with c3:
            input_comments = st.number_input("Beklenen Yorum Sayısı:", min_value=0, value=50, step=5)
            
        if st.button("Seçili Model ile Tahmin Et"):
            active_model = trained_models[selected_model_name]
            pred = active_model.predict([[input_followers, input_comments]])
            st.success(f"**{selected_model_name}** Tahmini: **{int(pred[0]):,} Beğeni**")
    else:
        st.error("ML modellerini eğitmek için veri setinde 'Followers', 'Comments' ve 'Likes' sütunları bulunmalıdır.")

# TAB 3: VERİ TABLOSU
with tab3:
    st.subheader("Filtrelenmiş Veri Seti")
    st.dataframe(df_filtered)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 CSV Olarak İndir",
        data=csv,
        file_name='sosyal_medya_verisi.csv',
        mime='text/csv'
    )
