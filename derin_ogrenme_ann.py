import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

print("==================================================")
print("  YAPAY SİNİR AĞLARI (ANN) - İYİLEŞTİRİLMİŞ MODEL")
print("==================================================\n")

# 1. Veri Yükleme
df = pd.read_csv('Social_Media_Sentiment_Analysis_AI_Trends_2026.csv')

# 2. Mantıksal Duygu Sınıfı Oluşturma (Feature Engineering)
def determine_sentiment(row):
    if row['toxicity_score'] > 0.5 or row['sentiment_score'] < -0.2:
        return 'Negative'
    elif row['sentiment_score'] > 0.2 and row['like_count'] > 500:
        return 'Positive'
    else:
        return 'Neutral'

df['target_sentiment'] = df.apply(determine_sentiment, axis=1)

# 3. Özellikler ve Hedef Değişken
features = [
    'user_followers_count', 'post_length', 'like_count', 
    'comment_count', 'share_count', 'engagement_score', 
    'sentiment_score', 'toxicity_score'
]

X = df[features].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['target_sentiment'])

# 4. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Optimize Edilmiş Yapay Sinir Ağı (ANN)
ann_model = MLPClassifier(
    hidden_layer_sizes=(128, 64), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42
)

ann_model.fit(X_train_scaled, y_train)

# 6. Tahmin ve Değerlendirme
y_pred = ann_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"📌 İyileştirilmiş ANN Doğruluk Oranı (Accuracy): %{acc * 100:.2f}\n")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))