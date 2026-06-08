# Status Inference Engine Lokal

## Status Akhir

Inference engine lokal sudah siap untuk demo dan dokumentasi.

1. Engine inference lokal tersedia di `app/inference.py`.
2. Script validasi tersedia di `scripts/validate_inference.py`.
3. Dependency inference dan UI tersedia di:
   - `requirements-inference.txt`
   - `requirements-ui.txt`
4. Streamlit UI tersedia di `app/streamlit_app.py`.
5. Static GitHub Pages preview tersedia di `docs/index.html`.
6. Theme Streamlit disiapkan di `.streamlit/config.toml` agar screenshot lebih stabil.

## Artifact Lokal

Engine memakai artifact lokal berikut:

- `hasil/scam_hybrid_indobert_bilstm/hybrid_indobert_bilstm_scam_final.pth`
- `hasil/scam_hybrid_indobert_bilstm/indobert_classifier_scam_baseline.pth`
- `hasil/scam_hybrid_indobert_bilstm/tokenizer/tokenizer.json`
- `hasil/scam_hybrid_indobert_bilstm/preprocessing_config.json`
- `hasil/scam_hybrid_indobert_bilstm/test_split.csv`

File `.pth` dan `.zip` tidak disarankan masuk GitHub biasa karena ukurannya besar.

## Hasil Validasi Hybrid

Validasi lokal Hybrid berhasil dijalankan di CPU.

- Accuracy: `0.978261`
- Precision: `0.978261`
- Recall: `0.978261`
- F1-score: `0.978261`
- Confusion: `TP=45`, `TN=45`, `FP=1`, `FN=1`

Report validasi:

- `hasil/INFERENCE_VALIDATION_HYBRID.md`
- `hasil/inference_validation_hybrid.json`

## Cara Menjalankan Validasi

```bash
.venv/bin/python scripts/validate_inference.py --model hybrid --device cpu
```

## Cara Menjalankan UI Streamlit

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Jika port `8501` bentrok:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

## Hal Penting GitHub Pages

`docs/index.html` adalah halaman static untuk GitHub Pages. Prediksi model asli tetap
berjalan melalui Streamlit/Python karena GitHub Pages tidak menjalankan checkpoint PyTorch
`.pth`.
