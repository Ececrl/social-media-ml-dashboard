import pandas as pd
import numpy as np

print("==================================================")
print("  FAZ 1: ZAMAN SERİSİ ENTEGRASYONU VE FEATURE MÜHENDİSLİĞİ")
print("==================================================\n")

# 1. Temizlenmiş Veriyi Yükleme
df = pd.read_csv('temizlenmis_veri.csv')
df['posted_datetime'] = pd.to_datetime(df['posted_datetime'])

print(f"[1/4] Temizlenmiş veri yüklendi. Toplam Gönderi Sayısı: {len(df)}")

# 2. Günlük Seviyede Agregasyon (Günlük Metriklerin Hesaplanması)
df.set_index('posted_datetime', inplace=True)

gunluk_df = df.resample('D').agg({
    'engagement_score': 'mean',
    'sentiment_score': 'mean',
    'like_count': 'sum',
    'comment_count': 'sum',
    'share_count': 'sum',
    'post_id': 'count'  # Günlük toplam gönderi sayısı
}).rename(columns={'post_id': 'gunluk_post_sayisi'}).reset_index()

print("[2/4] Veri günlük bazda agregasyon işleminden geçirildi.")

# 3. Gecikmeli Değişkenler (Lag Features) Türetme
# 'shift(k)' fonksiyonu veriyi k gün geçmişe kaydırır.
gunluk_df['Lag_1_Etkilesim'] = gunluk_df['engagement_score'].shift(1)
gunluk_df['Lag_2_Etkilesim'] = gunluk_df['engagement_score'].shift(2)
gunluk_df['Lag_7_Etkilesim'] = gunluk_df['engagement_score'].shift(7)

gunluk_df['Lag_1_Duygu'] = gunluk_df['sentiment_score'].shift(1)

# 4. Hareketli İstatistikler (Rolling Window Features) Türetme
# Son 7 ve 14 günün ortalamaları ve standart sapmaları
gunluk_df['Rolling_7_Etkilesim_Ort'] = gunluk_df['engagement_score'].rolling(window=7).mean()
gunluk_df['Rolling_7_Etkilesim_Std'] = gunluk_df['engagement_score'].rolling(window=7).std()

gunluk_df['Rolling_14_Etkilesim_Ort'] = gunluk_df['engagement_score'].rolling(window=14).mean()
gunluk_df['Rolling_7_Duygu_Ort'] = gunluk_df['sentiment_score'].rolling(window=7).mean()

print("[3/4] Lag (Gecikme) ve Rolling (Hareketli İstatistik) özellikleri başarıyla oluşturuldu.")

# 5. Sonuçları Kaydetme
gunluk_df.to_csv('zaman_entegreli_veri.csv', index=False)
print("\n[4/4] İşlem Tamamlandı! Oluşturulan veri seti: 'zaman_entegreli_veri.csv'")

print("\n--- Türetilen Tablonun İlk 10 Satırı (Örnek) ---")
print(gunluk_df[['posted_datetime', 'engagement_score', 'Lag_1_Etkilesim', 'Rolling_7_Etkilesim_Ort']].head(10))