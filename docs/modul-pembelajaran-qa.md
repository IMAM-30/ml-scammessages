# Modul Pembelajaran Q&A

## Q&A 2026-06-20

### 1. Cell 3 standardisasi kolom dan label

**Pertanyaan:**  
Pada cell 3 standardisasi, `text column candidates` itu apa, apakah kayak `Pesan`, `pesan`, dll itu berasal dari dalam dataset jika iya dataset yang mana, begitu juga dengan `label column candidates`, `positive label` dan `negative`.

**Jawaban:**  
`TEXT_COLUMN_CANDIDATES` dan `LABEL_COLUMN_CANDIDATES` adalah daftar kemungkinan nama kolom, bukan isi data. Di dataset repo ini, file yang terbaca adalah `dataset/sms_spam_indo.csv` dengan kolom `Pesan` dan `Kategori`.

`Pesan` cocok sebagai kolom teks. `Kategori` cocok sebagai kolom label. `POSITIVE_LABELS` dipakai untuk mengubah label seperti `spam/scam` menjadi `1`, sedangkan `NEGATIVE_LABELS` mengubah label seperti `ham/normal` menjadi `0`.

Di dataset ini label aslinya adalah `spam` dan `ham`, jadi `spam -> 1`, `ham -> 0`.

**Status pemahaman:** Sudah dipahami.

### 2. Fungsi kandidat yang tidak ada di dataset

**Pertanyaan:**  
Terus yang tidak ada di dataset itu gunanya apa? Kenapa dilampirkan di sana, untuk apa, jika tidak ada gunanya nanti dihapus manual.

**Jawaban:**  
Yang tidak ada di dataset berfungsi sebagai cadangan nama kolom atau label jika notebook dipakai untuk dataset lain. Untuk dataset saat ini, yang terpakai hanya `Pesan`, `Kategori`, `spam`, dan `ham`.

Kandidat lain tidak mengubah hasil karena kode hanya memilih yang cocok dengan dataset. Kalau proyek memang hanya memakai `dataset/sms_spam_indo.csv`, kandidat lain boleh dihapus manual, tapi tetap aman jika dibiarkan.

**Status pemahaman:** Sudah dipahami.

### 3. Kesimpulan kolom teks, kolom label, dan kelas positif-negatif

**Pertanyaan:**  
Jadi itu dapat dikatakan text dan label column yang diambil di dataset yang kita miliki kan? Serta label itu berasal dari label yang ada di dataset kita hanya dua: spam berarti positive pesan scam dan ham berarti negative penipuan.

**Jawaban:**  
Ya, `text_col` dan `label_col` diambil dari kolom yang ada di dataset. Pada dataset ini, `text_col = Pesan` dan `label_col = Kategori`.

Label juga berasal dari isi kolom `Kategori`. Di dataset ini hanya ada dua label utama: `spam` sebagai kelas positif atau scam/spam, dan `ham` sebagai kelas negatif atau normal/bukan penipuan.

**Status pemahaman:** Sudah dipahami.

### 4. Cara kerja fungsi `detect_column`

**Pertanyaan:**  
Pada `def detect_column` itu mendeteksi column yang ada kan setelah itu diubah menjadi huruf kecil, bagaimana pemahaman saya, coba juga jelaskan bagian def itu.

**Jawaban:**  
Pemahamannya hampir benar. Fungsi `detect_column` memang mendeteksi nama kolom yang ada di dataset, tetapi kolom aslinya tidak benar-benar diubah menjadi huruf kecil. Huruf kecil hanya dipakai untuk mencocokkan nama kolom agar tidak sensitif terhadap besar-kecil huruf.

`lower_map` membuat peta seperti `pesan -> Pesan` dan `kategori -> Kategori`. Lalu fungsi mengecek kandidat satu per satu. Jika ada yang sama persis, langsung dipakai. Jika tidak, dicek versi huruf kecilnya. Hasil akhirnya tetap mengembalikan nama kolom asli dari dataset, misalnya `Pesan` atau `Kategori`.

**Status pemahaman:** Sudah dipahami.

### 5. Pencocokan huruf besar-kecil pada `detect_column`

**Pertanyaan:**  
Jadi ini akan memeriksa bahwa kalau di `TEXT_COLUMN_CANDIDATES` ditulis `pesan`, sedangkan di dataset itu `Pesan`, maka tetap terbaca karena `lower_map` membuat kombinasi huruf besar-kecil sama saja asal katanya sesuai.

**Jawaban:**  
Ya, benar. Selama katanya sama dan hanya beda huruf besar-kecil, fungsi tetap bisa menemukan kolomnya. Contoh: kandidat `pesan` tetap cocok dengan kolom dataset `Pesan` karena keduanya dibandingkan dalam bentuk lowercase.

Yang tidak ditoleransi adalah salah kata yang berbeda, misalnya `psan` atau `pesann`, karena itu bukan cuma beda huruf besar-kecil.

**Status pemahaman:** Sudah dipahami.

### 6. Cara kerja `normalize_label`

**Pertanyaan:**  
`def normalize_label` pada bagian ini mengubah setiap positive label menjadi atau menghasilkan `1` dan negative jadi `0`? Setelah itu `1` dan `0` didefinisikan menjadi bilangan `int`?

**Jawaban:**  
Ya. `normalize_label` membaca label asli dari dataset, lalu mengubah label positif seperti `spam` menjadi `1`, dan label negatif seperti `ham` menjadi `0`.

Setelah itu, kolom `label` dibuat menjadi tipe integer dengan `base_df["label"] = base_df["label"].astype(int)`. Jadi hasil akhirnya label model berbentuk angka `0` dan `1`.

**Status pemahaman:** Sudah dipahami.

### 7. Asal `base_df.shape` dan tampilan 5 data

**Pertanyaan:**  
Ada `base_df.shape`, shape itu dibuat di mana sehingga menghasilkan `(1143, 3)`? Dan bagian mana yang menampilkan visual data dengan panjang 5?

**Jawaban:**  
`shape` bukan dibuat manual, tetapi properti bawaan DataFrame pandas. Nilai `(1143, 3)` muncul saat kode menjalankan `print("Clean base shape:", base_df.shape)`. Artinya ada 1143 baris data dan 3 kolom: `text`, `label_raw`, dan `label`.

Jumlah baris 1143 berasal dari data yang sudah dipilih dan dibersihkan dari `raw_df`. Tampilan 5 data pertama berasal dari `display(base_df.head())`, karena `.head()` secara default menampilkan 5 baris pertama.

**Status pemahaman:** Sudah dipahami.

### 8. Fungsi `value_counts().sort_index()`

**Pertanyaan:**  
Bagian ini dari mana: `value_counts().sort_index()`?

**Jawaban:**  
Itu method bawaan pandas. `value_counts()` menghitung jumlah kemunculan tiap label, misalnya berapa data `0` dan berapa data `1`. `sort_index()` mengurutkan hasil berdasarkan nilai labelnya, jadi `0` tampil sebelum `1`.

Pada kode ini dipakai di `base_df["label"].value_counts().sort_index()` untuk menampilkan distribusi label setelah label diubah menjadi angka.

**Status pemahaman:** Sudah dipahami.

### 9. Asal nilai dari `value_counts().sort_index()`

**Pertanyaan:**  
Maksudnya dari mana nilai atau hasil yang didapatkan oleh `value_counts().sort_index()`?

**Jawaban:**  
Nilainya berasal dari kolom `base_df["label"]`, yaitu label yang sudah diubah menjadi angka. Pada dataset ini, `ham` sudah menjadi `0` dan `spam` sudah menjadi `1`.

`value_counts()` menghitung jumlah data untuk tiap angka itu. Jadi hasilnya berasal dari jumlah baris label `0` dan jumlah baris label `1` di `base_df`.

**Status pemahaman:** Sudah dipahami.

### 10. Penjelasan khusus split pertama

**Pertanyaan:**  
Jelaskan split pertama dulu. `base_df` adalah jumlah data, lalu `train_val_df` nilainya dihitung di mana? `test_df` nilainya dari mana? Semua perhitungannya dari mana, sedangkan isi split hanya `base_df` dan `test_size`. Apakah keduanya output?

**Jawaban:**  
Pada split pertama, `train_val_df` dan `test_df` belum punya nilai sebelum baris itu dijalankan. Keduanya dibuat dari hasil output fungsi `train_test_split(...)`.

`base_df` adalah data input yang akan dibagi. Jumlah barisnya 1143. `test_size` berisi nilai `0.08`, artinya fungsi mengambil sekitar 8% dari `base_df` untuk output kedua, yaitu `test_df`.

Perhitungannya dilakukan otomatis di dalam fungsi `train_test_split`. Secara konsep: 1143 dikali 0.08 menjadi sekitar 91.44, lalu dibulatkan oleh fungsi menjadi 92 data test. Sisa data 1143 - 92 menjadi 1051 data, dan itu masuk ke `train_val_df`.

Jadi pada baris `train_val_df, test_df = train_test_split(...)`, fungsi menghasilkan dua output. Output pertama masuk ke `train_val_df`, output kedua masuk ke `test_df`.

**Status pemahaman:** Sudah dipahami.

### 11. Penjelasan khusus split kedua

**Pertanyaan:**  
Coba jelaskan split kedua dengan gaya penjelasan split pertama. Penjelasan sebelumnya yang membingungkan dibuang saja.

**Jawaban:**  
Pada split kedua, data yang dipakai bukan `base_df` lagi, tetapi `train_val_df`. `train_val_df` adalah sisa data dari split pertama, jumlahnya 1051 data.

Nilai `val_ratio_from_remaining` dipakai sebagai ukuran output kedua. Nilainya sekitar `0.1304`, artinya fungsi mengambil sekitar 13.04% dari `train_val_df` untuk output kedua, yaitu `val_df`.

Perhitungannya dilakukan otomatis di dalam `train_test_split`. Secara konsep: 1051 dikali 0.1304 menjadi sekitar 137.08, lalu dibulatkan oleh fungsi menjadi 138 data validation. Sisa data 1051 - 138 menjadi 913 data, dan itu masuk ke `train_df`.

Jadi pada baris `train_df, val_df = train_test_split(...)`, fungsi menghasilkan dua output. Output pertama masuk ke `train_df`, output kedua masuk ke `val_df`.

**Status pemahaman:** Sudah dipahami.

### 12. Maksud konfigurasi pada perhitungan split

**Pertanyaan:**  
Konfigurasi itu maksudnya apa pada perhitungan split?

**Jawaban:**  
Konfigurasi adalah tempat menyimpan nilai pengaturan yang dipakai kode. Di notebook ini namanya `CONFIG`, bentuknya seperti kamus berisi angka dan pilihan utama, misalnya `train_size`, `val_size`, `test_size`, `seed`, dan nama model.

Pada perhitungan split, angka seperti `0.80`, `0.12`, dan `0.08` tidak ditulis langsung di cell 4, tetapi diambil dari `CONFIG`. Jadi `CONFIG["train_size"]` berarti ambil nilai train dari konfigurasi, yaitu `0.80`.

Konfigurasi bukan hasil dari dataset. Itu nilai yang ditentukan oleh pembuat notebook sebagai rencana pembagian data. Kalau ingin mengubah rasio split, yang diubah cukup nilai di `CONFIG`.

**Status pemahaman:** Sudah dipahami.

### 13. Maksud `val_ratio_from_remaining`

**Pertanyaan:**  
Oke sudah paham, tapi mundur sedikit: mengapa atau apa maksud dari `val_ratio` yang dihitung di awal, yang memakai nilai validation dan train?

**Jawaban:**  
`val_ratio_from_remaining` dipakai untuk mengubah target validation dari persen total data menjadi persen dari data sisa setelah test diambil.

Target awal adalah train 80%, validation 12%, dan test 8%. Setelah test 8% diambil, data yang tersisa tinggal train + validation, yaitu 80% + 12% = 92%.

Karena split kedua hanya melihat sisa 92% itu, maka validation 12% harus dihitung terhadap 92%. Rumusnya menjadi `0.12 / 0.92 = 0.1304`. Artinya ambil 13.04% dari data sisa agar hasilnya setara dengan 12% dari total data awal.

Nilai train 0.80 dipakai di penyebut untuk menunjukkan bahwa data sisa setelah test adalah train + validation, yaitu `0.80 + 0.12`.

**Status pemahaman:** Sudah dipahami.

### 14. Koreksi pemahaman data sisa 92%

**Pertanyaan:**  
Jadi data untuk validation dan train itu hanya 92%, bukan 100%, karena test sudah diambil. Maka yang sisa dibagi itu hanya nilai train untuk mendapatkan nilai validation? Validation itu sekitar 0.130?

**Jawaban:**  
Benar bahwa data untuk train dan validation hanya berasal dari sisa 92%, bukan dari 100% data awal, karena 8% sudah dipisahkan menjadi test.

Yang dibagi pada split kedua bukan hanya train, tetapi gabungan calon train dan validation, yaitu `train_val_df`. Dari `train_val_df` itu, sekitar 13.04% diambil menjadi `val_df`, lalu sisanya menjadi `train_df`.

Jadi `0.1304` bukan 13.04% dari total data awal. Itu adalah 13.04% dari sisa 92%, supaya hasil akhirnya tetap sekitar 12% validation dari total data awal.

**Status pemahaman:** Sudah dipahami.

### 15. Rangkuman pemahaman alur split data

**Pertanyaan:**  
Pemahaman saya: pertama ambil 8% untuk test, lalu tersisa 92% untuk train dan validation. Setelah itu dihitung berapa persen validation dari sisa 92%. Split pertama memakai data full untuk mendapatkan `test_df`, lalu sisanya menjadi `train_val_df`. Split kedua memakai `train_val_df`, dikalikan dengan nilai validation yang sudah dihitung, hasilnya menjadi `val_df`, lalu sisanya menjadi `train_df`. Apakah begitu?

**Jawaban:**  
Ya, pemahaman itu sudah benar. Koreksi kecilnya: perhitungan `val_ratio` tidak benar-benar menunggu split pertama selesai, karena nilainya sudah bisa dihitung dari konfigurasi `0.12 / 0.92`. Tetapi secara konsep, benar bahwa nilai itu dipakai untuk mengambil validation dari sisa data 92%.

Split pertama memakai `base_df` sebagai data penuh. Dari 1143 data, sekitar 8% menjadi `test_df` sebanyak 92 data, dan sisanya menjadi `train_val_df` sebanyak 1051 data.

Split kedua memakai `train_val_df`. Dari 1051 data itu, sekitar 13.04% menjadi `val_df` sebanyak 138 data, dan sisanya menjadi `train_df` sebanyak 913 data.

**Status pemahaman:** Sudah dipahami.

## Catatan Lanjutan

Sesi berikutnya lanjut dari `hasil/sms-spam.ipynb` cell 5.
