# GitHub Pages `/docs`

Folder ini adalah versi static untuk live preview GitHub Pages.

## Cara Pakai

1. Masukkan folder `docs/` ke repository.
2. Di GitHub: Settings -> Pages.
3. Source: branch utama, folder `/docs`.
4. Buka URL GitHub Pages yang diberikan GitHub.

## Hal Penting

GitHub Pages hanya static HTML/CSS/JS. Model PyTorch `.pth` tidak dijalankan di sini.
Prediksi model asli tetap memakai:

```bash
streamlit run app/streamlit_app.py
```

Setelah dependency inference berhasil terpasang:

```bash
.venv/bin/python -m pip install -r requirements-ui.txt
.venv/bin/python -m streamlit run app/streamlit_app.py
```
