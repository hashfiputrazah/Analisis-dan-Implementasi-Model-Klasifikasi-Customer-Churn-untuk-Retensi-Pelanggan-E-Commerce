# Analisis-dan-Implementasi-Model-Klasifikasi-Customer-Churn-untuk-Retensi-Pelanggan-E-Commerce

# E-Commerce Customer Churn Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://backend-fooddelivery-ds6zsfc8jmd42rdqg8gbem.streamlit.app/)
[![Looker Studio](https://img.shields.io/badge/Looker%20Studio-Dashboard-blue)](https://lookerstudio.google.com/reporting/34e8d006-5644-4846-b675-a0fad70e8c7c)

Proyek machine learning end-to-end untuk memprediksi customer churn pada platform e-commerce menggunakan teknik klasifikasi dan analisis interpretable dengan SHAP.

---

## 📋 Daftar Isi

- [Gambaran Proyek](#gambaran-proyek)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Hasil Model](#hasil-model)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Deployment](#deployment)
- [Rekomendasi Bisnis](#rekomendasi-bisnis)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Penulis](#penulis)

---

## 🎯 Gambaran Proyek

Proyek ini merupakan capstone project Modul 3 yang bertujuan membangun model machine learning untuk memprediksi probabilitas customer churn pada perusahaan e-commerce. Model ini memungkinkan tim marketing untuk secara proaktif menargetkan pelanggan berisiko tinggi dengan strategi retensi yang efektif.

### Tujuan Utama
- Membangun model klasifikasi biner yang dapat memprediksi customer churn dengan akurasi tinggi
- Mengidentifikasi faktor-faktor utama yang mempengaruhi keputusan pelanggan untuk churn
- Memberikan insight actionable untuk strategi retensi pelanggan

---

## 💼 Business Problem

### Context
Perusahaan e-commerce menghadapi tantangan dalam mempertahankan pelanggan. Customer churn menyebabkan kerugian pendapatan yang signifikan, dan biaya akuisisi pelanggan baru jauh lebih mahal dibanding mempertahankan pelanggan existing.

### Stakeholder
- **Divisi Big Data**: Bertanggung jawab membangun, mengevaluasi, dan deploy model
- **Tim Marketing & Customer Retention**: Pengguna utama model untuk merancang kampanye retensi

### Goals
1. **Primary**: Membangun model ML yang memprediksi probabilitas customer churn
2. **Secondary**: Mengidentifikasi variabel utama yang paling berpengaruh terhadap churn

### Metric Selection
- **Primary Metric**: **ROC-AUC** - untuk evaluasi performa model secara keseluruhan
- **Key Business Metric**: **Recall** - untuk meminimalkan False Negative (pelanggan churn yang terlewat)

**Alasan**: Konsekuensi False Negative (kehilangan pelanggan) lebih merugikan dibanding False Positive (promosi terbuang ke pelanggan loyal).

---

## 📊 Dataset

### Sumber Data
Dataset E-Commerce Customer Churn dengan **3,941 records** (setelah cleaning: **3,270 records**)

### Data Dictionary

| Fitur | Tipe Data | Deskripsi |
|-------|-----------|-----------|
| `Tenure` | Float | Masa berlangganan pelanggan (bulan) |
| `WarehouseToHome` | Float | Jarak gudang ke rumah pelanggan (km) |
| `NumberOfDeviceRegistered` | Integer | Jumlah perangkat terdaftar |
| `PreferedOrderCat` | Text | Kategori pesanan favorit |
| `SatisfactionScore` | Integer | Skor kepuasan (1-5) |
| `MaritalStatus` | Text | Status pernikahan |
| `NumberOfAddress` | Integer | Jumlah alamat terdaftar |
| `Complain` | Integer | Keluhan dalam sebulan terakhir (1=Ya, 0=Tidak) |
| `DaySinceLastOrder` | Float | Hari sejak pesanan terakhir |
| `CashbackAmount` | Float | Rata-rata cashback (Rp) |
| `Churn` | Integer | **Target variable** (1=Churn, 0=Tidak) |

### Karakteristik Dataset
- **Imbalanced Data**: Churn=1 hanya **16.33%** dari total data
- **Missing Values**: Terdapat pada kolom `DaySinceLastOrder` (5.5%), `Tenure` (4.9%), dan `WarehouseToHome` (4.1%)
- **Data Duplikat**: 671 records (telah dihapus)

---

## 🔬 Metodologi

### 1. Data Understanding & Cleaning
- ✅ Handling duplikat data (671 records removed)
- ✅ Missing value imputation (mean strategy untuk numerik)
- ✅ Outlier detection (retained karena menggunakan tree-based model)
- ✅ EDA dan visualisasi distribusi data

### 2. Feature Engineering
- Pipeline preprocessing dengan `ColumnTransformer`
- **Numerical features**: Imputation (mean) + StandardScaler (untuk model linear)
- **Categorical features**: Imputation (most_frequent) + OneHotEncoder

### 3. Handling Imbalanced Data
Teknik yang diuji:
- **Random Over Sampling (ROS)** ⭐ **Terbaik**
- Random Under Sampling (RUS)
- SMOTENC

### 4. Model Benchmarking
9 algoritma diuji:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- **XGBoost**
- **LightGBM**
- **CatBoost** ⭐ **Terbaik**

### 5. Hyperparameter Tuning
- Menggunakan `RandomizedSearchCV` dengan 10-Fold Cross-Validation
- Parameter tuning untuk CatBoost:
  - `n_estimators`: [100, 200, 300, 400, 500, 1000]
  - `max_depth`: [3, 6, 9]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2, 0.3, 'auto']
  - `l2_leaf_reg`: [1, 3, 5, 7, 9]

---

## 🏆 Hasil Model

### Model Terbaik: CatBoost + ROS (Tuned)

#### Performa pada Test Set

| Metric | Baseline (Imbalanced) | Baseline (ROS) | **Tuned (ROS)** |
|--------|----------------------|----------------|-----------------|
| **ROC-AUC** | 0.9646 | 0.9661 | **0.9634** |
| **Recall (Churn)** | 0.7664 | 0.9252 | **0.7944** |
| **Precision (Churn)** | 0.8119 | 0.7021 | **0.7265** |
| **Accuracy** | 0.9327 | 0.9235 | **0.9174** |

#### Classification Report (Tuned Model)
```
              precision    recall  f1-score   support

           0       0.96      0.94      0.95       547
           1       0.73      0.79      0.76       107

    accuracy                           0.92       654
   macro avg       0.84      0.87      0.85       654
weighted avg       0.92      0.92      0.92       654
```

### 🔍 Feature Importance (SHAP Analysis)

**Top 3 Fitur Paling Berpengaruh:**

1. **Tenure** (⬇️ Nilai rendah = Churn tinggi)
   - Pelanggan baru (tenure rendah) sangat berisiko churn
   
2. **Complain** (⬆️ Complain = 1 → Churn tinggi)
   - Pelanggan dengan keluhan memiliki probabilitas churn jauh lebih tinggi
   
3. **DaySinceLastOrder** (⬆️ Nilai tinggi = Churn tinggi)
   - Pelanggan tidak aktif lama → cenderung churn

---

## 💰 Cost-Benefit Analysis

### Asumsi Biaya
- **Biaya Promosi/Intervensi**: Rp 1.000.000 per pelanggan
- **Biaya Kehilangan Pelanggan**: Rp 3.000.000 per pelanggan (3x biaya promosi)

### Hasil Analisis (Test Set: 107 pelanggan churn)

| Skenario | Total Biaya | Penghematan |
|----------|-------------|-------------|
| **Tanpa Model** (Tidak ada tindakan) | Rp 321.000.000 | - |
| **Dengan Model Tuned** | Rp 183.000.000 | **Rp 138.000.000** |

**Model berhasil menyelamatkan 85 dari 107 pelanggan churn (79.4%)!** 🎯

---

## 🛠️ Instalasi

### Prerequisites
- Python 3.8+
- pip

### Clone Repository
```bash
git clone https://github.com/username/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Utama
```
streamlit==1.x.x
pandas==2.x.x
numpy==2.x.x
scikit-learn==1.x.x
catboost==1.x.x
xgboost==2.x.x
lightgbm==4.x.x
imbalanced-learn==0.x.x
shap==0.x.x
matplotlib==3.x.x
seaborn==0.x.x
```

---

## 🚀 Cara Penggunaan

### 1. Training Model (Optional)
Jalankan notebook Jupyter untuk melatih ulang model:
```bash
jupyter notebook "Analisis dan Implementasi Model Klasifikasi Customer Churn.ipynb"
```

### 2. Menjalankan Aplikasi Streamlit Lokal
```bash
streamlit run app.py
```
Aplikasi akan terbuka di browser pada `http://localhost:8501`

### 3. Input Data Pelanggan
Masukkan 10 fitur pelanggan:
- Tenure (slider)
- Jarak Gudang ke Rumah (slider)
- Jumlah Perangkat Terdaftar (number input)
- Skor Kepuasan (slider 1-5)
- Jumlah Alamat (number input)
- Ada Keluhan? (dropdown)
- Hari Sejak Order Terakhir (slider)
- Jumlah Cashback (slider)
- Kategori Pesanan Favorit (dropdown)
- Status Pernikahan (dropdown)

### 4. Hasil Prediksi
Model akan memberikan:
- **Probabilitas Churn** (%)
- **Prediksi Kelas** (Churn/Tidak Churn)
- **Rekomendasi Aksi** berdasarkan hasil prediksi

---

## 🌐 Deployment

### Streamlit Community Cloud
Aplikasi telah dideploy dan dapat diakses di:
**🔗 [E-Commerce Churn Prediction App](https://backend-fooddelivery-ds6zsfc8jmd42rdqg8gbem.streamlit.app/)**

### Looker Studio Dashboard
Visualisasi data dan insight bisnis:
**📊 [Dashboard Looker Studio](https://lookerstudio.google.com/reporting/34e8d006-5644-4846-b675-a0fad70e8c7c)**

### Deploy ke Streamlit Cloud (Manual)
1. Fork repository ini
2. Pastikan file berikut ada:
   - `app.py`
   - `churn_prediction_model.pkl`
   - `requirements.txt`
3. Login ke [Streamlit Cloud](https://streamlit.io/cloud)
4. Klik "New app" dan pilih repository Anda
5. Set main file path: `app.py`
6. Deploy!

---

## 📈 Rekomendasi Bisnis

### 1. Rekomendasi untuk Data
- ✅ Kumpulkan lebih banyak data pelanggan churn untuk menyeimbangkan dataset
- ✅ Tambahkan fitur baru:
  - `AverageOrderValue`: Rata-rata nilai belanja
  - `FrequencyOfPurchase`: Frekuensi pembelian
  - `PromoUsage`: Penggunaan kode promo
  - `CustomerServiceInteractions`: Interaksi dengan CS

### 2. Rekomendasi untuk Model
- ✅ Lakukan tuning threshold prediksi (sesuaikan dengan trade-off bisnis)
- ✅ Iterasi hyperparameter tuning yang lebih ekstensif
- ✅ Implementasi feature selection untuk model yang lebih efisien

### 3. Rekomendasi Aksi Bisnis Berdasarkan Fitur Kunci

#### 🔴 **Tenure Rendah** (Pelanggan Baru 0-3 bulan)
- Buat program onboarding khusus
- Berikan insentif untuk pembelian kedua dan ketiga
- Follow-up personal setelah pembelian pertama

#### 🔴 **Complain = Ya**
- **Prioritas tertinggi!** Tangani keluhan dengan cepat
- Berikan voucher "permintaan maaf"
- Follow-up untuk memastikan kepuasan

#### 🔴 **DaySinceLastOrder Tinggi** (>15 hari)
- Kampanye win-back: "Kami Merindukanmu!"
- Kirim diskon khusus via email/push notification
- Personalisasi rekomendasi produk

#### 🟡 **MaritalStatus = Single**
- Analisis lebih lanjut: sensitivitas harga vs pesaing
- Sesuaikan promosi untuk segmen ini
- A/B testing strategi retensi

### 4. Implementasi Model
- Deploy model ke sistem CRM
- Otomasi scoring churn harian/mingguan
- Buat dashboard monitoring performa model
- Set up alert untuk pelanggan high-risk

---

## 🧰 Teknologi yang Digunakan

### Machine Learning
- **Scikit-learn**: Preprocessing, pipeline, evaluasi
- **CatBoost**: Model klasifikasi terbaik
- **XGBoost & LightGBM**: Benchmarking
- **Imbalanced-learn**: Handling imbalanced data (ROS/RUS/SMOTE)
- **SHAP**: Model interpretability

### Data Analysis & Visualization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computation
- **Matplotlib & Seaborn**: Visualisasi data
- **Missingno**: Missing value visualization

### Deployment
- **Streamlit**: Web application framework
- **Pickle**: Model serialization
- **Looker Studio**: Business intelligence dashboard

---

## 👨‍💻 Penulis

**Hashfi Putraza Hikmat**  
Kelas: JCDSBDGAM-09  
Capstone Project - Modul 3

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dalam program Job Connector Data Science Purwadhika.

---

## 🙏 Acknowledgments

- Dataset: E-Commerce Customer Churn Dataset
- Purwadhika Digital Technology School
- Instruktur dan mentor program JCDS

---

## 📞 Kontak

Untuk pertanyaan atau kolaborasi, silakan hubungi:
- 📧 Email: hashfiputrazah@gmail.com
- 💼 LinkedIn: [https://www.linkedin.com/in/hashfi-putraza-hikmat-151463195/]
- 🐙 GitHub: [https://github.com/hashfiputrazah]

---

**⭐ Jika proyek ini bermanfaat, jangan lupa berikan star!**
