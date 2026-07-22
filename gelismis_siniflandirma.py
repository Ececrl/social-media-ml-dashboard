import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("==================================================")
print("  GELİŞMİŞ SINIFLANDIRMA: LDA VE GLM (LOGISTIC REGRESSION)")
print("==================================================\n")

# 1. Veri Yükleme ve Hazırlık
df = pd.read_csv('temizlenmis_veri.csv')

features = [
    'user_followers_count', 'post_length', 'like_count', 
    'comment_count', 'share_count', 'engagement_score', 
    'toxicity_score', 'sarcasm_detected', 'spam_flag'
]

X = df[features].copy()
X['sarcasm_detected'] = X['sarcasm_detected'].astype(int)
X['spam_flag'] = X['spam_flag'].astype(int)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment_label'])

# Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Modeller
models = {
    "Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
    "GLM / Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"📌 --- {name} ---")
    print(f"Doğruluk Oranı (Accuracy): %{acc * 100:.2f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("-" * 50)