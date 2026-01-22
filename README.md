# Prediksi Prevalensi HIV Menggunakan XGBoost Regression

## Pendahuluan

Proyek ini bertujuan untuk menerapkan metode *machine learning*, khususnya **Extreme Gradient Boosting (XGBoost) Regression**, dalam memprediksi prevalensi HIV berdasarkan data numerik terstruktur.  
Pengembangan proyek ini dilakukan dalam konteks mata kuliah **Kecerdasan Komputasional**, dengan fokus pada pemodelan prediktif dan evaluasi kinerja model.

Pendekatan yang digunakan diharapkan dapat memberikan gambaran mengenai pemanfaatan algoritma pembelajaran mesin dalam bidang kesehatan masyarakat, khususnya dalam analisis data prevalensi penyakit.

## Tujuan

Tujuan dari pengembangan proyek ini adalah sebagai berikut:

1. Mengimplementasikan algoritma XGBoost Regression untuk prediksi data prevalensi HIV.
2. Melakukan analisis terhadap pola data menggunakan pendekatan pembelajaran mesin.
3. Mengevaluasi kinerja model regresi berdasarkan metrik evaluasi yang relevan.
4. Menerapkan konsep *reproducibility* melalui penyimpanan model terlatih.

## Dataset

Dataset yang digunakan dalam proyek ini berupa data numerik yang berkaitan dengan prevalensi HIV.  
Tahapan praproses data meliputi:

- Pemeriksaan dan penanganan data kosong (*missing values*)
- Pemilihan fitur yang relevan
- Penyesuaian format data agar sesuai dengan kebutuhan model

Berkas dataset:

- `hiv_prevalence.csv`

## Metodologi

Tahapan pengembangan model dilakukan melalui langkah-langkah berikut:

1. Memuat dan mengeksplorasi data awal
2. Melakukan praproses data
3. Melatih model menggunakan algoritma XGBoost Regression
4. Melakukan evaluasi hasil prediksi
5. Menyimpan model terlatih untuk penggunaan selanjutnya

Model yang telah dilatih disimpan dalam format `.pkl` guna mendukung penggunaan ulang dan konsistensi hasil.

## Implementasi Sistem

Struktur berkas utama dalam proyek ini adalah sebagai berikut:

- `XGBoost_Regression.ipynb`  
  Notebook Jupyter yang berisi proses eksplorasi data, pelatihan model, dan evaluasi hasil.

- `prediksihiv.py`  
  Berkas Python untuk melakukan prediksi menggunakan model yang telah dilatih.

- `modelXGboost.pkl`  
  Model hasil pelatihan yang telah diserialisasi.

- `requirements.txt`  
  Daftar pustaka Python yang digunakan dalam proyek.

## Teknologi yang Digunakan

- Python
- XGBoost
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

## Instalasi

Untuk menginstal seluruh dependensi yang dibutuhkan, jalankan perintah berikut:

```bash
pip install -r requirements.txt
