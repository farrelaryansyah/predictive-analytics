# Laporan Proyek Machine Learning - Prediksi Harga Motor Bekas

#### Disusun oleh : Mohamad Farrel Aryansyah

## Domain Proyek

Sepeda motor merupakan salah satu moda transportasi utama di Indonesia karena harganya yang terjangkau, efisiensi bahan bakar, dan fleksibilitas dalam menghadapi kemacetan. Karena harga motor baru terus meningkat, banyak masyarakat beralih ke pasar motor bekas sebagai solusi yang lebih ekonomis. Namun demikian, penentuan harga motor bekas sering kali tidak konsisten karena dipengaruhi oleh banyak variabel seperti merk, tahun pembuatan, jarak tempuh, tipe varian, serta kondisi dokumen resmi seperti STNK dan BPKB.

Ketidakpastian dalam menentukan harga motor bekas ini menjadi tantangan bagi konsumen maupun penjual. Konsumen kesulitan membandingkan harga yang wajar untuk motor dengan kondisi tertentu, sementara penjual berisiko memberikan harga yang tidak kompetitif di pasar. Oleh karena itu, perusahaan dan platform jual beli motor bekas perlu membangun sistem prediksi harga yang akurat dan objektif. Model prediktif berbasis machine learning mampu mengolah pola dari data historis dan fitur kendaraan untuk memberikan estimasi harga yang lebih masuk akal.

Beberapa penelitian telah menunjukkan bahwa pendekatan machine learning efektif dalam memprediksi harga kendaraan. Misalnya, studi oleh Putra *et al.* \[1] menunjukkan bahwa model seperti Random Forest dan Decision Tree mampu menghasilkan akurasi tinggi dalam prediksi harga mobil bekas, dengan Random Forest menjadi model paling unggul berdasarkan nilai R² dan MAE. Di sisi lain, laporan dari OLX Autos dan MarkPlus, Inc. \[2] juga mengungkap bahwa transparansi harga merupakan faktor kunci dalam kepercayaan konsumen terhadap transaksi kendaraan bekas. Oleh sebab itu, model prediksi harga motor bekas sangat relevan untuk meningkatkan efisiensi pasar dan pengalaman pengguna.

## Business Understanding

Penentuan harga motor bekas kerap kali bersifat subjektif, karena bergantung pada intuisi penjual dan minimnya standar penilaian yang jelas. Akibatnya, pembeli mengalami kesulitan dalam menilai kewajaran harga berdasarkan kondisi kendaraan yang ditawarkan. Ketidaksesuaian ini dapat menyebabkan ketidakseimbangan pasar serta menurunkan kepercayaan konsumen.

Untuk menjawab permasalahan tersebut, proyek ini mengembangkan model prediktif berbasis machine learning yang mampu memperkirakan harga motor bekas secara lebih akurat. Estimasi dilakukan berdasarkan fitur-fitur seperti merk, tahun produksi, jarak tempuh, tipe atau varian, warna, serta kelengkapan dokumen seperti STNK, BPKB, dan buku servis. Model ini diharapkan dapat meningkatkan transparansi dan efisiensi dalam proses transaksi jual beli.

### Problem Statements

* Bagaimana memprediksi harga motor bekas secara akurat berdasarkan data historis kendaraan?
* Algoritma machine learning mana yang paling optimal untuk memodelkan hubungan antara fitur kendaraan dan harga pasar?

### Goals

* Mengembangkan model regresi untuk memprediksi harga motor bekas berdasarkan fitur-fitur kendaraan.
* Membandingkan performa beberapa algoritma regresi untuk menentukan model terbaik berdasarkan evaluasi metrik.

### Solution Statements

* Menerapkan tiga algoritma machine learning: **Decision Tree Regressor**, **Random Forest Regressor**, dan **Gradient Boosting Regressor**.
* Melakukan **hyperparameter tuning** dengan **GridSearchCV** untuk meningkatkan akurasi model.
* Mengevaluasi model menggunakan metrik yang terukur, yaitu **MAE**, **RMSE**, dan **R² Score** pada data uji.
* Memilih model dengan performa terbaik untuk diimplementasikan dalam sistem prediksi harga motor bekas.

## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan hasil *web scraping* dari situs [mofe.co.id](https://mofe.co.id), yang merupakan platform penjualan motor bekas di Indonesia. Dataset difokuskan hanya pada motor bekas **merk Honda**, karena merupakan merk yang paling dominan dan tersedia dalam jumlah yang cukup besar di situs tersebut. Dataset dapat diunduh melalui tautan berikut:

🔗 [Link Dataset (Google Drive)](https://drive.google.com/uc?id=1-y6XXsoTLwaqOevLGh7DCxZGbTyaSBwO)

### Informasi Dataset

- Dataset memiliki format CSV.  
- Dataset terdiri dari 1.636 sampel motor bekas dengan merk Honda.  
- Dataset memiliki 2 fitur bertipe numerik (`int64`) dan 8 fitur bertipe `object`.  
- Terdapat missing value/null pada kolom **Tipe / Varian**.

### Variabel-variabel pada Dataset

| Fitur             | Deskripsi                                                                 |
|------------------|---------------------------------------------------------------------------|
| **Harga**         | Harga jual motor bekas (target variabel)                                 |
| **Merk & Model**  | Gabungan nama merk (Honda) dan model motor                               |
| **Tahun**         | Tahun pembuatan kendaraan                                                 |
| **Tipe / Varian** | Tipe motor seperti CBS, ABS, atau varian lainnya                         |
| **Jenis Transmisi** | Jenis transmisi motor (Matic, Manual, Cub)                              |
| **Warna**         | Warna kendaraan                                                |
| **Jarak Tempuh**  | Total kilometer yang telah ditempuh motor                                 |
| **STNK**          | Status kepemilikan dokumen STNK (*Tersedia* / *Tidak Tersedia*)           |
| **BPKB**          | Status kepemilikan dokumen BPKB (*Tersedia* / *Tidak Tersedia*)           |
| **Buku Servis**   | Ketersediaan buku servis kendaraan (*Tersedia* / *Tidak Tersedia*)        |

### Eksplorasi Awal

- Terdapat missing value pada kolom **Tipe / Varian**.
- Kolom **Harga** ditulis dalam format teks dengan awalan “Rp” dan pemisah ribuan berupa titik (“.”).
- Kolom **Jarak Tempuh** berisi data dalam format string dengan satuan “KM” dan pemisah ribuan.
- Kolom **Merk & Model** mencantumkan nama merk “Honda” diikuti dengan model motor, dipisahkan oleh koma.
- Nilai pada kolom **STNK**, **BPKB**, dan **Buku Servis** berupa teks kategori “Tersedia” dan “Tidak Tersedia”.

## Univariate Analysis

Univariate Analysis dilakukan untuk memahami distribusi fitur numerik dan kategorikal secara individu.

### Analisis sebaran pada setiap fitur numerik

![sebaran_numerik](https://github.com/user-attachments/assets/d2a96d35-785e-4569-8611-f7572596b766)

**Gambar 1**. Chart Analisis sebaran pada setiap fitur numerik.

Visualisasi **Gambar 1** di atas menunjukkan histogram dari tiga fitur utama: **Tahun**, **Jarak Tempuh**, dan **Harga**.

- **Tahun**: Sebagian besar motor bekas dalam dataset diproduksi antara tahun 2021 hingga 2024. Hal ini mencerminkan bahwa motor yang dijual tergolong baru atau usia muda, yang berpengaruh pada harga pasar.

- **Jarak Tempuh**: Distribusi jarak tempuh menunjukkan bahwa mayoritas motor memiliki jarak tempuh di bawah 50.000 KM. Nilainya sangat bervariasi namun condong ke arah nilai rendah (right-skewed), dengan beberapa outlier di atas 100.000 KM.

- **Harga**: Sebaran harga motor bekas berkisar antara Rp10 juta hingga Rp30 juta, dengan puncak distribusi di sekitar Rp15–18 juta. Pola distribusi menunjukkan skew ke kanan, yang mengindikasikan bahwa sebagian besar motor berada di rentang harga menengah ke bawah.

### Analisis Sebaran Fitur Kategorikal

![sebaran_kategorikal](https://github.com/user-attachments/assets/e7eb3ab3-e74a-4606-8d22-4e6e697e6100)
 
**Gambar 2**. Chart Analisis sebaran pada setiap fitur kategorikal.

Visualisasi **Gambar 2** menunjukkan countplot (jumlah kategori) untuk seluruh fitur kategorikal.

- **Merk & Model**: Model **BeAT** mendominasi dengan lebih dari 450 unit. Model lainnya seperti **Vario 125**, **Scoopy**, dan **Genio** juga cukup signifikan. Distribusi model sangat timpang, menunjukkan dominasi beberapa model populer.

- **Tipe / Varian**: Tipe **CBS** mendominasi, diikuti oleh **ISS**, **CBS ISS**, dan varian kombinasi lainnya. Sebagian besar data terfokus pada 3–5 varian utama, sementara sisanya hanya muncul dalam jumlah kecil.

- **Jenis Transmisi**: Mayoritas motor memiliki **transmisi matic**, menunjukkan bahwa matic adalah pilihan utama konsumen. Transmisi **sport** dan **cub** hanya sedikit.

- **Warna**: Warna **Hitam**, **Putih**, dan **Merah** paling umum. Warna seperti **Coklat**, **Ungu**, dan **Orange** jauh lebih jarang ditemukan, yang bisa mengindikasikan permintaan lebih rendah atau ketersediaan yang langka.

- **STNK** dan **BPKB**: Kedua dokumen ini **tersedia pada hampir seluruh motor**, yang penting untuk legalitas dan nilai jual kembali. Unit tanpa STNK atau BPKB sangat sedikit.

- **Buku Servis**: Uniknya, mayoritas motor **tidak memiliki buku servis**, yang mungkin disebabkan oleh kelalaian dalam pencatatan atau kurangnya perhatian pada histori perawatan oleh pemilik sebelumnya.

## Multivariate Analysis

Multivariate Analysis bertujuan untuk memahami hubungan antara dua atau lebih fitur, khususnya untuk melihat bagaimana fitur-fitur input memengaruhi **Harga** sebagai target.

### Korelasi Antar Fitur Numerik

![heatmap_numerik](https://github.com/user-attachments/assets/120ec06b-a417-4061-b383-af96fb558688)

**Gambar 3**. Heatmap korelasi antar fitur numerik: `Tahun`, `Jarak Tempuh`, dan `Harga`.

Berdasarkan **Gambar 3**, dapat dilihat:

- **Tahun dan Jarak Tempuh** memiliki korelasi negatif yang kuat (**-0.65**), artinya motor yang lebih tua cenderung memiliki jarak tempuh lebih tinggi.
- **Jarak Tempuh dan Harga** memiliki korelasi negatif (**-0.31**), menunjukkan bahwa semakin besar jarak tempuh, harga cenderung turun.
- **Tahun dan Harga** memiliki korelasi positif (**0.42**), yang artinya motor lebih baru cenderung dihargai lebih mahal.

### Pairplot Fitur Numerik

![pairplot_numerik](https://github.com/user-attachments/assets/9996f324-b4bc-481b-b88c-27983e914f5b)

**Gambar 4**. Pairplot fitur numerik: `Tahun`, `Jarak Tempuh`, dan `Harga`.

Pairplot pada **Gambar 4** di atas memperlihatkan hubungan visual antara fitur numerik:

- **Tahun vs Harga**: Terlihat pola korelasi positif yang cukup jelas — semakin baru tahunnya, semakin tinggi harga motor.
- **Jarak Tempuh vs Harga**: Terdapat korelasi negatif — motor dengan jarak tempuh rendah cenderung memiliki harga lebih tinggi.
- **Tahun vs Jarak Tempuh**: Pola sebaran menunjukkan bahwa motor keluaran lama memiliki jarak tempuh lebih tinggi, mendukung hasil korelasi sebelumnya.

Visualisasi ini memperkuat insight numerik bahwa `Tahun` dan `Jarak Tempuh` berperan penting dalam menentukan `Harga`, dan dapat dijadikan sebagai fitur utama dalam proses pemodelan.

### Distribusi Harga Berdasarkan Fitur Kategorikal

![boxplot_kategorikal_harga](https://github.com/user-attachments/assets/68abcabe-f7ff-40e0-83db-a8915ae83745)

**Gambar 5**. Boxplot distribusi harga berdasarkan fitur kategorikal.

Visualisasi **Gambar 5** menunjukkan hubungan antara harga dan beberapa fitur kategorikal:

- **Merk & Model**: Model seperti **ADV150**, **PCX160**, dan **CB150R** memiliki median harga lebih tinggi dibanding model seperti **Revo** atau **Blade125**. Hal ini menunjukkan bahwa tipe motor merupakan faktor signifikan dalam penentuan harga.

- **Tipe / Varian**: Varian **ABS**, **CBS ISS**, dan **ISS** cenderung dihargai lebih tinggi karena fitur tambahan yang ditawarkan.

- **Jenis Transmisi**: Motor **Sport** dan **Matic** menunjukkan median harga lebih tinggi dibanding **Cub**, yang biasanya lebih ekonomis.

- **Warna**: Warna seperti **Hijau**, **Merah**, dan **Hitam** memiliki persebaran harga lebih tinggi, meskipun pengaruh warna terhadap harga secara umum tidak terlalu signifikan.

- **STNK dan BPKB**: Motor dengan **STNK** dan **BPKB tersedia** menunjukkan median harga lebih tinggi. Kelengkapan dokumen secara langsung meningkatkan nilai jual motor.

- **Buku Servis**: Motor dengan **buku servis tersedia** sedikit lebih tinggi secara median, meskipun tidak sebesar pengaruh STNK/BPKB. Hal ini menunjukkan bahwa histori servis tidak menjadi faktor utama, tetapi tetap menambah nilai kepercayaan pembeli.

## Data Preparation

Tahapan data preparation dilakukan untuk memastikan data siap digunakan dalam proses modeling machine learning. Proses ini melibatkan pembersihan data, penanganan nilai kosong, transformasi tipe data, dan encoding fitur kategorikal.

### 1. Pembersihan Data

- **Harga**: Kolom harga awalnya berupa string dalam format `'Rp xx.xxx.xxx'`. Data ini dibersihkan dengan menghapus simbol `'Rp'`, titik pemisah ribuan, lalu dikonversi menjadi integer.
  
- **Jarak Tempuh**: Sama seperti harga, kolom ini awalnya dalam format `'xx.xxx KM'`. Dilakukan penghapusan `'KM'` dan titik, lalu dikonversi ke integer.
  
- **Merk & Model**: Kolom ini berisi gabungan merek dan model motor, misalnya 'Honda, ADV150'. Untuk menyederhanakan nilai pada kolom ini, dilakukan penghapusan string `'Honda,'` dan spasi di sekitarnya, sehingga menjadi `'ADV150'`.

### 2. Penanganan Missing Values

- Kolom **Tipe / Varian** memiliki nilai kosong (NaN). Untuk mengisinya:
  - Digunakan pendekatan berbasis grup: nilai kosong diisi dengan **modus (mode)** berdasarkan kelompok `Merk & Model`.
  - Jika tidak ditemukan mode (karena hanya 1 sampel), maka nilai diisi dengan string `'NaN'`.

### 3. Encoding Fitur Kategorikal

Fitur-fitur kategorikal tidak bisa langsung digunakan dalam model numerik. Oleh karena itu, dilakukan proses encoding menggunakan `LabelEncoder` dari scikit-learn:

- Fitur yang diencode meliputi:
  - `Merk & Model`
  - `Tipe / Varian`
  - `Jenis Transmisi`
  - `Warna`
  - `STNK`
  - `BPKB`
  - `Buku Servis`

Setiap kolom dikodekan menjadi nilai numerik diskrit berdasarkan urutan label.

### 4. Pembagian Dataset

Setelah proses encoding, data dipisahkan menjadi **fitur (X)** dan **target (y)**:
- `X` berisi semua fitur kecuali kolom `Harga`
- `y` adalah kolom `Harga` sebagai variabel target

Dataset kemudian dibagi menjadi **data latih (train)** dan **data uji (test)** dengan rasio 80:20 menggunakan `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Modeling

**Algoritma**

  Penelitian ini melakukan pemodelan dengan 3 algoritma regresi, yaitu **Decision Tree**, **Random Forest**, dan **Gradient Boosting**.

  - **Decision Tree**

    Decision Tree Regressor bekerja dengan membagi dataset ke dalam cabang berdasarkan fitur yang paling mengurangi variansi (impurity). Proyek ini menggunakan [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) dan dilatih dengan `X_train` dan `y_train`. Parameter yang digunakan antara lain:

    - `max_depth` = Kedalaman maksimum tree.
    - `min_samples_split` = Jumlah minimum sampel untuk memisahkan node.
    - `random_state` = Seed untuk reprodusibilitas.

  - **Random Forest**

    Random Forest adalah metode ensemble learning yang membangun banyak decision tree dan menggabungkan hasilnya secara rata-rata. Proyek ini menggunakan [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Model ini dilatih dengan `X_train` dan `y_train`, dengan parameter:

    - `n_estimators` = Jumlah pohon yang dibangun.
    - `max_depth` = Kedalaman maksimum pohon.
    - `min_samples_split` = Minimum sampel untuk split.
    - `random_state` = Seed acak untuk hasil yang konsisten.

  - **Gradient Boosting**

    Gradient Boosting adalah metode boosting yang membangun model secara berurutan dan mengoreksi kesalahan model sebelumnya. Proyek ini menggunakan [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html), dengan parameter:

    - `n_estimators` = Jumlah estimator boosting.
    - `max_depth` = Kedalaman maksimum setiap pohon.
    - `learning_rate` = Ukuran kontribusi tiap model.
    - `random_state` = Seed acak untuk konsistensi.

**Hyperparameter Tuning (Grid Search)**

  Untuk meningkatkan performa model, dilakukan **Grid Search** untuk mencari kombinasi parameter terbaik. Teknik ini menggunakan validasi silang (`cv=3`) dan metrik `neg_mean_absolute_error`.

Berikut hasil tuning terbaik untuk masing-masing model:

| Model              | Best Parameters                                                               |
|-------------------|--------------------------------------------------------------------------------|
| Decision Tree      | `{'max_depth': 20, 'min_samples_split': 2}`                                   |
| Random Forest      | `{'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}`            |
| Gradient Boosting  | `{'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1}`                |

## Evaluation

### Metrik Evaluasi:

- **MAE (Mean Absolute Error)**: Rata-rata selisih absolut antara nilai prediksi dan nilai aktual. Metrik ini menunjukkan seberapa jauh prediksi dari nilai sebenarnya, tanpa memperhatikan arah kesalahan.
- **MSE (Mean Squared Error)**: Rata-rata kuadrat dari selisih antara nilai prediksi dan nilai aktual. Lebih sensitif terhadap error yang besar.
- **RMSE (Root Mean Squared Error)**: Akar dari MSE, memberikan skala yang sama dengan target (harga).
- **R² Score (Coefficient of Determination)**: Mengukur proporsi variasi pada target yang dapat dijelaskan oleh model. Nilai mendekati 1 menunjukkan model yang sangat baik.

### Hasil Evaluasi (Default vs Tuned)

| Model              | MAE Default | MAE Tuned | R² Default | R² Tuned |
|-------------------|-------------|-----------|------------|----------|
| Decision Tree      | 615.243  | Rp 616.081 | 0.9278     | 0.9272   |
| Random Forest      | 572.891  | Rp 568.904 | 0.9565     | 0.9566   |
| Gradient Boosting  | 823.975  | Rp 552.314 | 0.9473     | 0.9509   |

### Hasil MSE Train dan Test (dalam ribuan)

| Model              | MSE Train (Ribu) | MSE Test (Ribu) |
|-------------------|------------------|-----------------|
| Decision Tree      | 108.236.000      | 1.629.165.000   |
| Random Forest      | 233.624.000      | 971.403.000     |
| Gradient Boosting  | 107.684.000      | 1.099.065.000   |

> Gradient Boosting memiliki **train-test gap yang kecil**, menunjukkan generalisasi yang baik.

![perbandingan_mse](https://github.com/user-attachments/assets/3c5674b8-4fba-45f4-8208-9ff9f13aa7eb)

**Gambar 6**. Perbandingan MSE.

Visualisasi **Gambar 6** menampilkan perbandingan antara **Train MSE** dan **Test MSE** untuk masing-masing model:

- **Random Forest** memiliki selisih MSE yang besar antara data train dan test, namun tetap menunjukkan generalisasi yang lebih baik dibandingkan model lainnya.
- **Gradient Boosting** juga menunjukkan peningkatan MSE pada data test, tetapi performanya masih cukup stabil.
- **Decision Tree** menunjukkan MSE test paling tinggi, menandakan adanya **overfitting** yang signifikan terhadap data latih.

### Kesimpulan Evaluasi

- **Random Forest Regressor (Tuned)** memberikan performa paling konsisten dengan MAE dan R² terbaik.
- **Gradient Boosting Regressor (Tuned)** menghasilkan **MAE terendah** (`Rp 552.314`) dan **R² sangat tinggi** (`0.9509`), dengan performa yang sangat kompetitif.
- **Decision Tree Regressor** mengalami **sedikit penurunan performa setelah tuning**, baik dari sisi MAE, MSE, maupun R². Ini menunjukkan bahwa parameter default pada Decision Tree sudah cukup optimal untuk dataset ini.
- Seluruh model ensemble menunjukkan peningkatan performa yang signifikan setelah tuning, dan **Random Forest** dipilih sebagai model akhir karena keseimbangan antara akurasi tinggi dan kestabilan performa.

## Reference:

\[1] P. H. Putra, B. P. Azanuddin, dan Y. A. Dalimunthe, "Random forest and decision tree algorithms for car price prediction," *Jurnal Matematika Dan Ilmu Pengetahuan Alam LLDikti Wilayah 1 (JUMPA)*, vol. 3, no. 2, pp. 81–89, 2023.

\[2] OLX Autos & MarkPlus, Inc., "Consumer Behavior Report: Used Car Market in Indonesia," 2021.
