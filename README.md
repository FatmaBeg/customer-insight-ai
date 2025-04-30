
# 🧠 Customer Insight AI Platform (Northwind Edition)

Bu proje, **Northwind veritabanı** kullanılarak geliştirilen ve müşteri davranışlarını tahmin etmeye yönelik **üç farklı derin öğrenme modelini** kapsayan bir yapay zeka çözüm platformudur. Proje, her bir modeli bağımsız API'ler olarak servis eder ve veri bilimi araştırmalarına uygun olarak yapılandırılmıştır.

## 🚀 Amaçlar ve Modüller

### 1️⃣ Sipariş Verme Alışkanlığı Tahmini
Müşterinin geçmişteki sipariş verileri üzerinden, **önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini** tahmin eder.

- Özellikler: Toplam harcama, sipariş sayısı, ortalama sipariş büyüklüğü, son sipariş tarihi vb.
- Ar-Ge:
  - **Temporal Features** (mevsimsellik etkisi)
  - **Class Imbalance** çözümü (SMOTE, class weights)
  - **Data Augmentation**

### 2️⃣ Ürün İade Risk Skoru
Bir siparişin **iade edilme olasılığını** tahmin eder.

- Özellikler: İndirim oranı, ürün adedi, toplam harcama
- Ar-Ge:
  - **Cost-sensitive Learning**
  - **Explainable AI** (SHAP / LIME ile karar açıklaması)
  - Sahte etiketleme stratejisi (yüksek indirim + düşük harcama = yüksek risk)

### 3️⃣ Yeni Ürün Satın Alma Potansiyeli
Müşterinin geçmiş kategorik harcamalarına göre, yeni çıkan bir ürünü **satın alma olasılığını** tahmin eder.

- Özellikler: Ürün kategorilerine göre geçmiş harcamalar
- Ar-Ge:
  - **Neural Recommendation Systems**
  - **Multi-label Prediction**

## 🏗️ Proje Yapısı

```
customer_insight_ai/
├── app/              # FastAPI servisleri
├── pipeline/         # Model eğitimi ve ön işleme
├── data/             # Northwind veri dosyaları
├── models/           # Eğitilmiş modeller
├── notebooks/        # EDA ve araştırma not defterleri
├── requirements.txt  # Bağımlılıklar
└── README.md
```

## ⚙️ Kurulum

### 1. Ortamı Kur
```bash
git clone https://github.com/kullanici/customer-insight-ai.git
cd customer_insight_ai
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. Veri Kaynağını Hazırla
`data/raw/` klasörüne Northwind veritabanını (`.csv` veya `.sqlite`) yerleştirin.

## 🚉 API Kullanımı

```bash
uvicorn app.main:app --reload
```

Swagger arayüzü için:
```
http://127.0.0.1:8000/docs
```

### Örnek Endpoint'ler:

| Problem                      | Endpoint URL                     | Input Tipi              |
|-----------------------------|----------------------------------|--------------------------|
| Sipariş Tahmini             | `/predict-repeat-order/`         | `CustomerFeatures`       |
| İade Riski                  | `/predict-return-risk/`          | `OrderFeatures`          |
| Yeni Ürün Önerisi           | `/predict-new-product/`          | `PurchaseCategoryInput`  |

## 📊 Kullanılan Teknolojiler

- **Python 3.10+**
- **FastAPI**
- **TensorFlow / Keras**
- **scikit-learn, pandas, numpy**
- **imbalanced-learn (SMOTE)**
- **SHAP, LIME**
- **SQLite / PostgreSQL opsiyonlu**
- **joblib / pickle / h5**

## 🧪 Geliştirme / Araştırma

Her model için `pipeline/` klasörü altında ayrı eğitim, preprocessing ve veri analiz script'leri mevcuttur. Ek olarak `notebooks/` klasörü, her bir model için yapılmış EDA (Exploratory Data Analysis) çalışmalarını içerir.

## ✨ Gelecek Geliştirmeler

- 🎯 Otomatik model güncelleyici cron job'lar
- 📈 Model performans metrik API'leri (monitoring)
- 🔐 Kullanıcı oturum sistemi (JWT)

## 👩‍💻 Katkıda Bulunmak

Pull request'ler ve issue'lar açıktır. Her katkıyı memnuniyetle karşılıyoruz!

## 📝 Lisans

Bu proje MIT lisansı ile lisanslanmıştır.
