import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

print("==================================================")
print("  GELİŞMİŞ KÜMELEME: HİYERARŞİK VE GAUSS KARIŞIM MODELİ (GMM)")
print("==================================================\n")

# 1. Veri Yükleme ve Ölçekleme
df = pd.read_csv('temizlenmis_veri.csv')

features = [
    'user_followers_count', 'post_length', 'like_count', 
    'comment_count', 'share_count', 'engagement_score', 
    'sentiment_score', 'toxicity_score'
]

X_scaled = StandardScaler().fit_transform(df[features])

# 2. Hiyerarşik Kümeleme (Hierarchical Clustering)
hc = AgglomerativeClustering(n_clusters=3)
df['HC_Cluster'] = hc.fit_predict(X_scaled)
hc_sil = silhouette_score(X_scaled, df['HC_Cluster'])

print(f"📌 Hiyerarşik Kümeleme Silhouette Skoru : %{hc_sil * 100:.2f}")

# 3. Gauss Karışım Modeli (GMM - Olasılıksal Kümeleme)
gmm = GaussianMixture(n_components=3, random_state=42)
df['GMM_Cluster'] = gmm.fit_predict(X_scaled)
gmm_sil = silhouette_score(X_scaled, df['GMM_Cluster'])

print(f"📌 Gauss Karışım Modeli (GMM) Silhouette Skoru: %{gmm_sil * 100:.2f}")

# GMM ile her gözlemin kümelere ait olma olasılıkları
probs = gmm.predict_proba(X_scaled)
print("\n--- İlk 3 Gönderinin Kümeye Ait Olma Olasılıkları (GMM) ---")
for i in range(3):
    print(f"Gönderi {i+1}: Küme 0: %{probs[i][0]*100:.1f} | Küme 1: %{probs[i][1]*100:.1f} | Küme 2: %{probs[i][2]*100:.1f}")

df.to_csv('gelismis_kumeleme_sonuclari.csv', index=False)
print("\nSonuçlar 'gelismis_kumeleme_sonuclari.csv' olarak kaydedildi!")