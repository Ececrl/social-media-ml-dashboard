import os

print("==================================================")
print("       OTOMATİK RAPOR OLUŞTURUCU (MARKDOWN)")
print("==================================================\n")

markdown_content = """# 📊 Sosyal Medya Trend & Etkileşim Analizi - Makine Öğrenmesi Raporu

**Tarih:** 2026-07-22  
**Geliştiren:** İş Zekası (BI) Staj Takımı  
**Veri Seti:** `Social_Media_Sentiment_Analysis_AI_Trends_2026.csv`

---

## 📌 1. Proje Özeti ve Amaç
Bu çalışma, sosyal medya platformlarındaki etkileşim dinamiklerini ve duygu durumlarını analiz ederek **geleceğe dönük etkileşim skorlarını tahminlemek** ve **duygu durumlarını yüksek doğrulukla sınıflandırmak** amacıyla geliştirilmiştir.

Uçtan uca modüler bir boru hattı (pipeline) kurgulanmış; veri temizleme, zaman serisi özellik mühendisliği (feature engineering), regresyon modelleri, gelişmiş kümeleme/sınıflandırma ve Yapay Sinir Ağları (ANN) uygulanmıştır.

---

## 🛠️ 2. Metodoloji ve Özellik Mühendisliği (Feature Engineering)
- **Veri Ön İşleme:** Eksik veriler işlenmiş ve tipler doğrulanmıştır.
- **Zaman Serisi Entegrasyonu:** Geçmiş eğilimleri modele öğretebilmek amacıyla `Lag_1`, `Lag_2` (gecikmeli değişkenler) ve `Rolling_Mean_3` (3 günlük hareketli ortalama) öznitelikleri türetilmiştir.
- **Sınıflandırma Hedef Kurgusu:** Duygu, etkileşim ve toksisite skorları harmanlanarak `target_sentiment` mantıksal hedef değişkeni oluşturulmuştur.

---

## 📈 3. Regresyon Modelleri ve Performans Kıyaslaması
İleriye dönük etkileşim tahmini için 3 farklı regresyon algoritması eğitilmiş ve test seti üzerinde değerlendirilmiştir:

| Model | MAE | RMSE | $R^2$ Skoru |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **1337.72** | **1684.76** | **%94.47** |
| **Ridge Regression** | 1337.61 | 1689.39 | %94.44 |
| **Random Forest Regressor** | 2274.97 | 2907.92 | %83.51 |

> **Analiz Notu:** Zaman serisi özniteliklerinin eklenmesiyle birlikte **Linear Regression** modeli verideki doğrusal trendi mükemmel yakalayarak **%94.47 $R^2$** başarısı göstermiştir.

---

## 🧠 4. Derin Öğrenme (ANN) Sınıflandırma Performansı
Duygu ve etkileşim durumlarını sınıflandırmak amacıyla çok katmanlı bir **Yapay Sinir Ağı (MLPClassifier)** eğitilmiştir.

- **Mimari:** 2 Gizli Katman (128 ve 64 Nöron), ReLU Aktivasyonu, Adam Optimizer
- **Doğruluk Oranı (Accuracy):** **%96.00**

> **Analiz Notu:** Derin öğrenme modeli, mantıksal olarak kurgulanan hedef değişken üzerindeki karmaşık ve doğrusal olmayan desenleri başarıyla öğrenmiştir.

---

## 📁 5. Proje Dosya Yapısı ve Modüller
- `veri_temizleme.py`: Ham veri işleme ve temizleme.
- `zaman_serisi_entegrasyonu.py`: Lag ve Rolling öznitelik türetimi.
- `regresyon_modeli.py`: Regresyon algoritmaları ve `oznitelik_onemleri.png` grafik üretimi.
- `derin_ogrenme_ann.py`: %96 doğruluklu Yapay Sinir Ağı modeli.
- `gelismis_siniflandirma.py` & `gelismis_kumeleme.py`: LDA, GLM, Hierarchical & GMM analizleri.
- `rapor_olusturucu.py`: İşbu raporu otomatik üreten script.

---
*Rapor otomatik olarak sistem tarafından oluşturulmuştur.*
"""

with open("Regresyon_Model_Raporu.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("📝 Rapor başarıyla güncellendi: 'Regresyon_Model_Raporu.md'")