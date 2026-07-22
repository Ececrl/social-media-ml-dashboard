import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("==================================================")
print("   İLERİYE DÖNÜK ETKİLEŞİM TAHMİNİ (REGRESYON)")
print("==================================================\n")

# 1. Veri Yükleme
df = pd.read_csv('Social_Media_Sentiment_Analysis_AI_Trends_2026.csv')

# 2. Sektör Standardı Hedef Değişken (Trend ve Etkileşim Bileşimi)
np.random.seed(42)
df['base_engagement'] = (
    df['like_count'] * 1.0 + 
    df['comment_count'] * 2.0 + 
    df['share_count'] * 3.0
)

# Zaman Serisi Gecikmeli Değişkenler (Lag Features)
df['Lag_1'] = df['base_engagement'].shift(1)
df['Lag_2'] = df['base_engagement'].shift(2)
df['Rolling_Mean_3'] = df['base_engagement'].rolling(window=3).mean()

# Hedef Değişken: Gerçek Hayat Dalgalanmalı Etkileşim Skoru
df['target_engagement'] = df['base_engagement'] + np.random.normal(0, df['base_engagement'].std() * 0.25, len(df))

# Eksik Değerleri Temizleme
df = df.dropna().reset_index(drop=True)

# 3. Özellikler ve Hedef Değişken
features = [
    'user_followers_count', 'post_length', 'sentiment_score', 
    'toxicity_score', 'Lag_1', 'Lag_2', 'Rolling_Mean_3'
]

X = df[features]
y = df['target_engagement']

# 4. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Eğitimi ve Karşılaştırma
models = {
    "1. Linear Regression": LinearRegression(),
    "2. Ridge Regression": Ridge(alpha=1.0),
    "3. Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

print("📊 Model Performans Sonuçları:\n")
for name, model in models.items():
    if "Random Forest" in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {name} ---")
    print(f"  • MAE  : {mae:.2f}")
    print(f"  • RMSE : {rmse:.2f}")
    print(f"  • R²   : %{r2 * 100:.2f}\n")

# 6. Görselleştirme: Öznitelik Önem Düzeyleri
rf_model = models["3. Random Forest Regressor"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=importances, y=importances.index, hue=importances.index, palette="viridis", legend=False)
plt.title("Random Forest - Değişkenlerin Etkileşim Tahminindeki Önem Düzeyi")
plt.xlabel("Önem Skoru (Feature Importance)")
plt.ylabel("Değişkenler")
plt.tight_layout()

# Grafiği Kaydetme
plt.savefig("oznitelik_onemleri.png", dpi=300)
print("📈 Görsel grafik 'oznitelik_onemleri.png' olarak başarıyla kaydedildi!")