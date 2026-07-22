import pandas as pd

# 1. Veri setini yüklüyoruz
# (CSV dosyasının .py dosyasıyla aynı klasörde olduğundan emin ol)
df = pd.read_csv('Social_Media_Sentiment_Analysis_AI_Trends_2026.csv')

print("--- 1. Yükleme Başarılı ---")
print(f"Toplam Satır: {len(df)}, Toplam Sütun: {len(df.columns)}")

# 2. 'mentions' sütunundaki boş (NaN) değerleri 'None' ile dolduruyoruz
df['mentions'] = df['mentions'].fillna('None')

# 3. Tarih sütununu Python'ın zaman formatına çeviriyoruz
df['posted_datetime'] = pd.to_datetime(df['posted_datetime'])

# 4. Veriyi kronolojik olarak (eskiden yeniye) sıralıyoruz
df = df.sort_values('posted_datetime').reset_index(drop=True)

# 5. Temizlenmiş veriyi yeni bir CSV dosyası olarak kaydediyoruz
# (Böylece sonraki zaman serisi ve modelleme adımlarında bu temiz veriyi direkt kullanabiliriz)
df.to_csv('temizlenmis_veri.csv', index=False)

print("\n--- 2. Temizlik Tamamlandı ---")
print(f"Kalan boş (NaN) değer sayısı: {df.isnull().sum().sum()}")
print(f"En Eski Tarih: {df['posted_datetime'].min()}")
print(f"En Yeni Tarih: {df['posted_datetime'].max()}")
print("\nTemizlenmiş veri 'temizlenmis_veri.csv' olarak kaydedildi!")