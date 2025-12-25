# Road-Issues-Detection


## Deskripsi Proyek
Proyek ini merupakan implementasi sistem klasifikasi permasalahan infrastruktur jalan
berbasis citra digital menggunakan metode machine learning.

Tujuan utama dari proyek ini adalah membandingkan performa model **Neural Network
Non-Pretrained (CNN Baseline)** dengan dua model **Pretrained (Transfer Learning)**,
yaitu **DenseNet-201** dan **VGG-16**, dalam mengklasifikasikan berbagai jenis permasalahan
jalan seperti kerusakan rambu, jalan rusak, parkir ilegal, kebersihan lingkungan, dan lainnya.

Selain proses pelatihan dan evaluasi model, proyek ini juga dilengkapi dengan
website sederhana berbasis **Streamlit** untuk mendemonstrasikan hasil prediksi model
secara lokal.

---

## Dataset dan Preprocessing
Dataset yang digunakan adalah **Road Issues Detection Dataset** yang diperoleh dari
platform Kaggle (https://www.kaggle.com/datasets/programmerrdai/road-issues-detection-dataset). Dataset ini berisi lebih dari 9.660 citra RGB resolusi tinggi yang
dikategorikan ke dalam beberapa kelas permasalahan infrastruktur jalan.

### Kelas Dataset:
- Broken Road Sign Issues  
- Damaged Road Issues  
- Illegal Parking Issues  
- Littering Garbage on Public Places Issues  
- Mixed Issues  
- Pothole Issues  
- Vandalism Issues  

### Preprocessing Data:
Tahapan preprocessing yang dilakukan meliputi:
1. Pemindaian dataset menggunakan pendekatan **dataframe-based loader**
   (`flow_from_dataframe`) karena struktur folder dataset bersifat bertingkat.
2. Resize citra menjadi ukuran **64×64 piksel**.
3. Normalisasi nilai piksel dengan skala **1/255**.
4. Data augmentation pada data latih berupa rotasi, zoom, dan horizontal flip.
5. Pembagian data menjadi **80% data latih** dan **20% data validasi** secara stratified.

---

## Model yang Digunakan
Pada proyek ini digunakan tiga model machine learning, terdiri dari satu model
non-pretrained dan dua model pretrained.

### 1. CNN Baseline (Non-Pretrained)
Model CNN baseline dibangun dari awal tanpa menggunakan bobot pretrained.
Arsitektur model mengacu pada penelitian sebelumnya dengan susunan beberapa blok
Convolutional Neural Network yang terdiri dari:
- Convolution Layer
- Batch Normalization
- Average Pooling
- Dropout
- Fully Connected Layer

Model ini digunakan sebagai pembanding dasar terhadap model transfer learning.

---

### 2. DenseNet-201 (Pretrained)
DenseNet-201 merupakan model pretrained yang menggunakan bobot hasil pelatihan
pada dataset ImageNet. Pada implementasi ini:
- Layer convolutional utama freeze.
- Ditambahkan Global Average Pooling dan Fully Connected Layer pada bagian classifier.
- Model digunakan dengan pendekatan transfer learning untuk klasifikasi multi-kelas.

---

### 3. VGG-16 (Pretrained)
VGG-16 merupakan model pretrained dengan arsitektur CNN klasik yang memiliki
lapisan konvolusi berurutan.
Pada implementasi ini:
- Bobot pretrained ImageNet digunakan.
- Layer feature extractor freeze.
- Ditambahkan Fully Connected Layer untuk menyesuaikan jumlah kelas dataset.

---

## Hasil Evaluasi dan Analisis Perbandingan
Evaluasi model dilakukan menggunakan beberapa metrik klasifikasi, yaitu:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Hasil evaluasi menunjukkan bahwa model pretrained secara umum memiliki performa
yang lebih baik dibandingkan CNN baseline, terutama dalam mengenali pola kompleks
pada citra jalan.

### Tabel Perbandingan Performa Model

| Model             | Akurasi | Hasil Analisis Singkat |
|-------------------|---------|------------------------|
| CNN Baseline      | 87%     | Model non-pretrained mampu mengenali kelas dominan dengan baik, namun lemah pada kelas minoritas seperti Mixed Issues akibat keterbatasan data. |
| DenseNet-201      | 90%     | Model dengan performa terbaik, menunjukkan peningkatan akurasi dan kemampuan ekstraksi fitur yang lebih kuat dibandingkan CNN baseline, meskipun masih terbatas pada kelas minoritas. |
| VGG-16            | 89%     | Memberikan performa yang stabil dan mendekati DenseNet-201, namun kurang optimal pada beberapa kelas dengan jumlah data kecil. |



---

## Implementasi Sistem Website Menggunakan Streamlit
Sistem website sederhana dikembangkan menggunakan framework **Streamlit** untuk
mendemonstrasikan hasil klasifikasi model secara lokal.

Aplikasi Streamlit memungkinkan pengguna untuk:
1. Mengunggah citra jalan.
2. Memilih model klasifikasi yang digunakan (CNN Baseline, DenseNet-201, atau VGG-16).
3. Melihat hasil prediksi kelas dan nilai confidence dari model.

Model yang digunakan pada aplikasi Streamlit merupakan **model hasil pelatihan**
yang telah disimpan dalam format `.h5`, sehingga aplikasi hanya melakukan proses
inference tanpa pelatihan ulang.

---

## Panduan Menjalankan Sistem Website Secara Lokal

1. Clone repository GitHub proyek ini:
```bash
git clone https://github.com/InzaziNurmujaini/road-issues-detection.git
cd road-issues-detection
