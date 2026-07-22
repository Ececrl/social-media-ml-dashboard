import pandas as pd
import numpy as np

# Kümeleme ve Boyut İndirgeme Kütüphaneleri
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

print("==================================================")
print("  FAZ 4: GÖZETİMSİZ ÖĞRENME - KÜMELEME (CLUSTERING) & PCA")
print("==================================================\n")

# 1. Veri Setini Yükleme
df = pd.read_csv('temizlenmis_veri.csv')
print(f"[1/5] Veri yüklendi. Toplam Satır Sayısı: {len(df)}")

# 2. Kümeleme İçin Öznitelik Seçimi
features = [
    'user_followers_count', 'post_length', 'like_count', 
    'comment_count', 'share_count', 'engagement_score', 
    'sentiment_score', 'toxicity_score'
]

X = df[features].copy()

# Kümeleme algoritmaları mesafeye dayalı olduğu için ölçekleme şarttır!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[2/5] Öznitelikler seçildi ve StandardScaler ile ölçeklendi.")

# 3. Optimal Küme Sayısını Belirleme (Elbow Method & Silhouette Score)
print("\n[3/5] Optimal Küme Sayısı Analizi Yapılıyor...")
print(f"{'Küme Sayısı (K)':<18} | {'Inertia (SSE)':<18} | {'Silhouette Score':<18}")
print("-" * 60)

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, cluster_labels)
    
    print(f"{k:<18} | {inertia:<18.2f} | %{sil_score * 100:.2f}")

# En dengeli segmentasyon için K=3 seçip modeli eğitiyoruz
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# 4. PCA ile Boyut İndirgeme (8 Değişkeni 2 Boyuta Düşürme)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

print(f"\n[4/5] PCA Uygulandı. Açıklanan Toplam Varyans Oranı: %{sum(pca.explained_variance_ratio_) * 100:.2f}")

# 5. Kümelerin Profil Analizi (Segment İş Zekası / BI Raporu)
print("\n==================================================")
print("  KÜME (SEGMENT) PROFiLLERi VE ORTALAMALARI")
print("==================================================")

cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary.round(2).T)

# Sonuçları Kaydetme
df.to_csv('kumeleme_sonuclari.csv', index=False)
print("\n[5/5] Kümeleme sonuçları 'kumeleme_sonuclari.csv' olarak kaydedildi!")