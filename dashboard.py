import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
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
# 2. VERİ YÜKLEME (Zaman Entegreli Veri Seti)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("zaman_entegreli_veri.csv")
    # Eğer tarih sütunu varsa datetime formatına çevirelim
    for col in df.columns:
        if 'date' in col.lower() or 'tarih' in col.lower() or 'time' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='ignore')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR (KENAR ÇUBUĞU) FİLTRELERİ
# ---------------------------------------------------------
st.sidebar.title("🔍 Veri Filtreleme")

df_filtered = df.copy()

# Platform Filtresi (Varsa)
if 'Platform' in df.columns:
    platforms = df['Platform'].unique().tolist()
    selected_platforms = st.sidebar.multiselect("Platform Seçin:", options=platforms, default=platforms)
    if selected_platforms:
        df_filtered = df_filtered[df_filtered['Platform'].isin(selected_platforms)]

# İçerik Türü Filtresi (Varsa)
if 'Post_Type' in df.columns:
    post_types = df['Post_Type'].unique().tolist()
    selected_types = st.sidebar.multiselect("İçerik Türü Seçin:", options=post_types, default=post_types)
    if selected_types:
        df_filtered = df_filtered[df_filtered['Post_Type'].isin(selected_types)]

# ---------------------------------------------------------
# 4. KPI METRİK KARTLARI
# ---------------------------------------------------------
st.title("📱 Sosyal Medya Zaman Entegreli ML Dashboard")
st.markdown("Zaman serisi entegrasyonlu veri seti ve makine öğrenmesi modelleri karşılaştırması.")
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
tab1, tab2, tab3 = st.tabs(["📊 Zaman & Trend Analizi", "🤖 Regresyon Modelleri Karşılaştırması", "📋 Veri Tablosu"])

# TAB 1: GRAFİKLER & ZAMAN TRENDLERİ
with tab1:
    st.subheader("Gönderi & Zaman Serisi Analizleri")
    g1, g2 = st.columns(2)
    
    # Tarih/Zaman Sütununu Tespit Etme
    date_cols = [c for c in df_filtered.columns if 'date' in c.lower() or 'tarih' in c.lower() or 'time' in c.lower() or 'saat' in c.lower()]
    
    with g1:
        st.markdown("**Zaman İçindeki Beğeni Trendi**")
        if date_cols and 'Likes' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            # Zaman sıralı grafik
            df_sorted = df_filtered.sort_values(by=date_cols[0])
            sns.lineplot(data=df_sorted, x=date_cols[0], y='Likes', ax=ax, color='tab:blue')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        elif 'Platform' in df_filtered.columns and 'Likes' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_filtered, x='Platform', y='Likes', ax=ax, palette='Blues_d')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Trend grafiği için tarih/zaman sütunları kullanılıyor.")
            
    with g2:
        st.markdown("**Beğeni ve Yorum Sayısı İlişkisi**")
        if 'Likes' in df_filtered.columns and 'Comments' in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_filtered, x='Likes', y='Comments', hue='Platform' if 'Platform' in df_filtered.columns else None, ax=ax)
            st.pyplot(fig)

# TAB 2: REGRESYON MODELLERİ KARŞILAŞTIRMASI
with tab2:
    st.subheader("📊 Model Performans Sonuçları")
    
    # ML için yalnızca sayısal sütunları seçme
    numeric_df = df.select_dtypes(include=[np.number])
    
    if 'Likes' in numeric_df.columns and numeric_df.shape[1] > 1:
        X = numeric_df.drop(columns=['Likes'])
        y = numeric_df['Likes']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Karşılaştırılacak Modeller
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
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
            
            results.append({
                "Model": name, 
                "MAE": round(mae, 2), 
                "RMSE": round(rmse, 2),
                "R² (%)": f"%{r2 * 100:.2f}"
            })
            trained_models[name] = model
            
        results_df = pd.DataFrame(results)
        
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            st.markdown("#### 📋 Metrik Karşılaştırma Tablosu")
            st.dataframe(results_df, use_container_width=True)
            
        with col_chart:
            st.markdown("#### 📊 R² Performansı Kıyaslaması")
            r2_numeric = [r2_score(y_test, trained_models[m].predict(X_test)) for m in models.keys()]
            chart_df = pd.DataFrame({"Model": list(models.keys()), "R² Skoru": r2_numeric})
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=chart_df, x='Model', y='R² Skoru', ax=ax, palette='viridis')
            plt.xticks(rotation=20)
            st.pyplot(fig)
    else:
        st.error("ML modellerini eğitmek için sayısal sütunlar (ör. 'Likes') gerekiyor.")

# TAB 3: VERİ TABLOSU
with tab3:
    st.subheader("Filtrelenmiş Veri Seti")
    st.dataframe(df_filtered)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 CSV Olarak İndir",
        data=csv,
        file_name='zaman_entegreli_sosyal_medya_verisi.csv',
        mime='text/csv'
    )
