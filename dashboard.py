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
    
    # Tarih sütunlarını datetime formatına çevir
    for col in df.columns:
        if any(k in col.lower() for k in ['date', 'tarih', 'time', 'saat']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Veri yüklenirken hata oluştu: {e}")
    st.info("Lütfen GitHub ana dizininde 'zaman_entegreli_veri.csv' dosyasının bulunduğundan emin olun.")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR (KENAR ÇUBUĞU) FİLTRELERİ
# ---------------------------------------------------------
st.sidebar.title("🔍 Veri Filtreleme")

df_filtered = df.copy()

# Platform Filtresi (Eğer sütun mevcutsa)
platform_col = [c for c in df.columns if 'platform' in c.lower()]
if platform_col:
    p_col = platform_col[0]
    platforms = df[p_col].dropna().unique().tolist()
    selected_platforms = st.sidebar.multiselect("Platform Seçin:", options=platforms, default=platforms)
    if selected_platforms:
        df_filtered = df_filtered[df_filtered[p_col].isin(selected_platforms)]

# ---------------------------------------------------------
# 4. KPI METRİK KARTLARI
# ---------------------------------------------------------
st.title("📱 Sosyal Medya Performance & ML Dashboard")
st.markdown("Zaman entegreli veri seti ve makine öğrenmesi modelleri karşılaştırma paneli.")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

target_col = 'Likes' if 'Likes' in df_filtered.columns else df_filtered.select_dtypes(include=[np.number]).columns[0]

with col1:
    st.metric(label="📌 Toplam Gönderi", value=f"{len(df_filtered):,}")
with col2:
    avg_likes = df_filtered[target_col].mean() if target_col in df_filtered.columns else 0
    st.metric(label="❤️ Ort. Beğeni", value=f"{avg_likes:.1f}")
with col3:
    comments_col = [c for c in df_filtered.columns if 'comment' in c.lower() or 'yorum' in c.lower()]
    avg_comments = df_filtered[comments_col[0]].mean() if comments_col else 0
    st.metric(label="💬 Ort. Yorum", value=f"{avg_comments:.1f}")
with col4:
    max_likes = df_filtered[target_col].max() if target_col in df_filtered.columns else 0
    st.metric(label="🔥 Maksimum Beğeni", value=f"{max_likes:,}")

st.markdown("---")

# ---------------------------------------------------------
# 5. SEKMELER
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analiz & Grafikler", "🤖 Regresyon Modelleri Karşılaştırması", "📋 Veri Tablosu"])

# TAB 1: GRAFİKLER & TREND ANALİZİ
with tab1:
    st.subheader("Gönderi & Zaman Serisi Analizleri")
    g1, g2 = st.columns(2)
    
    date_cols = [c for c in df_filtered.columns if any(k in c.lower() for k in ['date', 'tarih', 'time'])]
    
    with g1:
        st.markdown("**Zaman İçindeki Beğeni Trendi**")
        if date_cols and target_col in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df_sorted = df_filtered.sort_values(by=date_cols[0])
            sns.lineplot(data=df_sorted, x=date_cols[0], y=target_col, ax=ax, color='tab:blue')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Zaman grafiği için tarih verisi işleniyor.")
            
    with g2:
        st.markdown("**Beğeni ve Etkileşim İlişkisi**")
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_filtered, x=numeric_cols[0], y=numeric_cols[1], ax=ax, color='purple')
            st.pyplot(fig)

# TAB 2: REGRESYON MODELLERİ KARŞILAŞTIRMASI
with tab2:
    st.subheader("📊 Model Performans Sonuçları")
    
    # ML Modelleri için Sayısal Sütunlar
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if target_col in numeric_df.columns and numeric_df.shape[1] > 1:
        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Görseldeki 3 Model
        models = {
            "1. Linear Regression": LinearRegression(),
            "2. Ridge Regression": Ridge(),
            "3. Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
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
            trained_models[name] = (model, r2)
            
        results_df = pd.DataFrame(results)
        
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            st.markdown("#### 📋 Metrik Karşılaştırma Tablosu")
            st.dataframe(results_df, use_container_width=True)
            
        with col_chart:
            st.markdown("#### 📊 R² Performansı Kıyaslaması")
            chart_data = pd.DataFrame({
                "Model": [name.split(". ")[1] for name in models.keys()],
                "R² Skoru": [trained_models[name][1] for name in models.keys()]
            })
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=chart_data, x='Model', y='R² Skoru', ax=ax, palette='viridis')
            plt.xticks(rotation=15)
            plt.ylim(0, 1.0)
            st.pyplot(fig)
    else:
        st.error("ML modellerini eğitmek için yeterli sayısal sütun bulunamadı.")

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
