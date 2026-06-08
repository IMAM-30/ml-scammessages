# Ringkasan Hasil Training UAS ML

## Status Artifact

- ZIP artifact: `hasil/scam_hybrid_indobert_bilstm.zip`
- Folder ekstrak: `hasil/scam_hybrid_indobert_bilstm`
- SHA256 ZIP: `e7d384f653b0653773601a2dda22e5d0d1bc8388c8d2f57d7aff6954e3d2d5f1`
- Checkpoint Hybrid: `hybrid_indobert_bilstm_scam_final.pth`
- Checkpoint IndoBERT baseline: `indobert_classifier_scam_baseline.pth`
- Tokenizer tersedia: `tokenizer/tokenizer.json`
- Konfigurasi preprocessing tersedia: `preprocessing_config.json`

## Dataset dan Split

- Dataset Kaggle: `/kaggle/input/notebooks/gevabriel/indonesian-sms-spam-detection-using-indobert/indo_spam.csv`
- Kolom teks: `Pesan`
- Kolom label: `Kategori`
- Label mapping: `{'normal_ham': 0, 'scam_spam': 1}`
- Train: 913 data
- Validation: 138 data
- Test: 92 data
- Distribusi test: {0: 46, 1: 46}

## Konfigurasi Training

- Model dasar: `indobenchmark/indobert-base-p2`
- Max length: 128
- Batch size: 16
- Learning rate: 2e-05
- Epoch IndoBERT: 6
- Epoch Hybrid: 10
- Early stopping patience: 3
- Early stopping min delta: 0.0001

## Metrik Final Test

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| IndoBERT Classifier | 0.989130 | 0.978723 | 1.000000 | 0.989247 | 1.000000 | 0.75 |
| Hybrid IndoBERT+BiLSTM | 0.978261 | 0.978261 | 0.978261 | 0.978261 | 0.993856 | 0.68 |
| SVM TF-IDF | 0.956522 | 0.937500 | 0.978261 | 0.957447 | N/A | 0.50 |

## Kesimpulan Teknis

1. Model terbaik berdasarkan F1 test adalah **IndoBERT Classifier** dengan F1 **0.989247** dan accuracy **0.989130**.
2. Model Hybrid IndoBERT+BiLSTM tetap sangat kuat dengan F1 **0.978261**, dan menjadi model utama sesuai proposal.
3. Hybrid mengungguli baseline klasik SVM TF-IDF: F1 Hybrid **0.978261** vs SVM **0.957447**.
4. IndoBERT fine-tuning murni mengungguli Hybrid pada split test ini: F1 IndoBERT **0.989247** vs Hybrid **0.978261**.
5. Untuk laporan, narasi yang aman: arsitektur Hybrid berhasil meningkatkan performa dibanding metode ML klasik, tetapi IndoBERT fine-tuning menjadi pembanding terbaik pada eksperimen ini.

## Kesiapan Untuk Inference/UI

Artifact sudah cukup untuk membangun inference lokal karena memiliki:

- checkpoint model `.pth`
- tokenizer
- preprocessing config dan slang dictionary
- threshold final
- split data dan hasil evaluasi

Dependency lokal terdeteksi:

- torch: False
- transformers: False
- streamlit: False

Langkah berikutnya sebelum UI: buat `app/inference.py`, load model Hybrid dari checkpoint, jalankan prediksi contoh scam/normal, lalu baru bangun Streamlit UI.
