import pandas as pd
import numpy as np

# Makine Öğrenmesi Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("==================================================")
print("  FAZ 2: GÖZETİMLİ ÖĞRENME - SINIFLANDIRMA (CLASSIFICATION)")
print("==================================================\n")

# 1. Veri Setini Yükleme
df = pd.read_csv('temizlenmis_veri.csv')
print(f"[1/5] Veri yüklendi. Toplam Satır Sayısı: {len(df)}")

# 2. Öznitelik (Feature) ve Hedef (Target) Seçimi
# Hedef değişkenimiz: sentiment_label (Positive, Negative, Neutral)
# Girdi değişkenlerimiz (X): Gönderi ve etkileşim özellikleri
features = [
    'user_followers_count', 'post_length', 'like_count', 
    'comment_count', 'share_count', 'engagement_score', 
    'toxicity_score', 'sarcasm_detected', 'spam_flag'
]

X = df[features].copy()
y = df['sentiment_label']

# Boole (True/False) olan sütunları sayısal 1/0 formatına çeviriyoruz
X['sarcasm_detected'] = X['sarcasm_detected'].astype(int)
X['spam_flag'] = X['spam_flag'].astype(int)

# Hedef değişkeni kategorikten sayısal etiketlere dönüştürme (Positive=2, Neutral=1, Negative=0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("[2/5] Öznitelikler (X) ve Hedef Değişken (y) ayrıştırıldı.")

# 3. Veriyi Eğitim (Train) ve Test Seti Olarak Ayırma (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# Mesafe tabanlı modeller (KNN ve SVM) için Veri Ölçekleme (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[3/5] Veri seti bölündü. Eğitim Seti: {len(X_train)}, Test Seti: {len(X_test)}")

# 4. Modellerin Tanımlanması ve Eğitilmesi
models = {
    "K-Nearest Neighbors (KNN)": (KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled),
    "Support Vector Machine (SVM)": (SVC(kernel='rbf', C=1.0, random_state=42), X_train_scaled, X_test_scaled),
    "Random Forest (Karar Ağaçları)": (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test)
}

print("\n==================================================")
print("  MODEL PERFORMANS KARŞILAŞTIRMALARI")
print("==================================================")

# 5. Modellerin Değerlendirilmesi
results = {}

for name, (model, train_data, test_data) in models.items():
    model.fit(train_data, y_train)
    y_pred = model.predict(test_data)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n📌 --- {name} ---")
    print(f"Doğruluk Oranı (Accuracy Score): %{acc * 100:.2f}")
    print("\nDetaylı Sınıflandırma Raporu (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print("Karmaşıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Random Forest Öznitelik Önem Düzeyleri (Feature Importance)
rf_model = models["Random Forest (Karar Ağaçları)"][0]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

print("\n🌳 --- Random Forest Öznitelik Önem Düzeyleri (Feature Importance) ---")
print("Duygu durumunu tahmin ederken en etkili olan değişkenler:")
for feature_name, importance in importances.items():
    print(f"  • {feature_name:<25}: %{importance * 100:.2f}")

print("\n[5/5] Sınıflandırma Fazı Başarıyla Tamamlandı!")