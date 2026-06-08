"""Streamlit UI for the Indonesian scam/spam detector."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.inference import DEFAULT_ARTIFACT_DIR, DependencyMissingError, load_engine


SUMMARY_CSV = PROJECT_ROOT / "hasil" / "ringkasan_metrik_final.csv"
ARTIFACT_MANIFEST = PROJECT_ROOT / "hasil" / "artifact_manifest.json"


st.set_page_config(
    page_title="Indonesian Scam Detector",
    page_icon="!",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_engine(model_type: str, device: str):
    return load_engine(DEFAULT_ARTIFACT_DIR, model_type=model_type, device=device)


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    if ARTIFACT_MANIFEST.exists():
        return json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def load_summary_rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.exists():
        return []
    import csv

    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_percent(value: str | float | int) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value) * 100:.2f}%"


def risk_badge(result) -> tuple[str, str]:
    if result.label_id == 1:
        if result.risk_level == "tinggi":
            return "Scam berisiko tinggi", "high"
        return "Scam perlu diwaspadai", "medium"
    return "Normal atau rendah risiko", "low"


def model_label(model_type: str) -> str:
    return "Hybrid IndoBERT+BiLSTM" if model_type == "hybrid" else "IndoBERT Classifier"


st.markdown(
    """
    <style>
    :root {
        --surface: #f5f5f7;
        --panel: rgba(255,255,255,0.92);
        --panel-strong: #ffffff;
        --ink: #1d1d1f;
        --muted: #6e6e73;
        --line: #d2d2d7;
        --accent: #0071e3;
        --ok: #0b7a4b;
        --warn: #9a5b00;
        --bad: #b42318;
        --shadow: 0 18px 45px rgba(0,0,0,0.07);
    }
    .stApp { background: var(--surface); color: var(--ink); }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1180px; }
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    header [data-testid="stToolbar"],
    header button[kind="header"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    header {
        height: 0 !important;
        visibility: hidden !important;
    }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink) !important; }
    .hero {
        box-shadow: var(--shadow);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 28px 32px;
        background: var(--panel);
        margin-bottom: 22px;
    }
    .hero-title {
        margin: 0 0 10px 0;
        color: var(--ink) !important;
        font-size: clamp(2.35rem, 4vw, 4.8rem);
        line-height: 1.02;
        font-weight: 800;
    }
    .subtle {
        color: var(--muted) !important;
        max-width: 760px;
        font-size: 1.04rem;
        line-height: 1.65;
    }
    .section-title {
        color: var(--ink) !important;
        font-size: 1.62rem;
        font-weight: 780;
        margin: 0 0 6px;
    }
    .section-help {
        color: var(--muted) !important;
        font-size: 0.96rem;
        line-height: 1.55;
        margin: 0 0 16px;
    }
    .metric-box {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        background: var(--panel-strong);
        box-shadow: 0 10px 24px rgba(0,0,0,0.045);
    }
    .metric-label { color: var(--muted) !important; font-size: 0.82rem; }
    .metric-value { color: var(--ink) !important; font-size: 1.45rem; font-weight: 750; }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 12px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid;
        margin: 4px 0 14px;
    }
    .badge.high { color: var(--bad); border-color: #f1b7bd; background: #fff1f2; }
    .badge.medium { color: var(--warn); border-color: #f0c78c; background: #fff7ea; }
    .badge.low { color: var(--ok); border-color: #a9dec5; background: #eefaf4; }
    .clean-text {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 13px 14px;
        background: var(--panel-strong);
        color: var(--ink) !important;
        line-height: 1.5;
    }
    div[data-testid="stSidebar"] {
        background: #fbfbfd;
        border-right: 1px solid var(--line);
    }
    div[data-testid="stButton"] button {
        border-radius: 999px;
        min-height: 44px;
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
        font-weight: 760;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 8px;
        border-color: var(--line);
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


manifest = load_manifest()
summary_rows = load_summary_rows()

with st.sidebar:
    st.header("Model")
    model_type = st.selectbox(
        "Checkpoint",
        options=["hybrid", "indobert"],
        format_func=model_label,
    )
    device = st.selectbox("Device", options=["cpu", "auto"], index=0)
    st.divider()
    st.caption("Artifact")
    st.success("Ready")
    st.caption("hasil/scam_hybrid_indobert_bilstm")
    if manifest:
        st.caption(f"Dataset test: {manifest.get('dataset', {}).get('split_counts', {}).get('test', '-')} data")

active_model_label = model_label(model_type)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">Deteksi Scam Bahasa Indonesia</div>
      <div class="subtle">Model aktif: <strong>{active_model_label}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.markdown(
        """
        <div class="section-title">Prediksi Pesan</div>
        """,
        unsafe_allow_html=True,
    )
    default_text = "Selamat Anda memenangkan hadiah 100 juta. Transfer biaya admin ke rekening 1234 sekarang."
    text = st.text_area("Teks pesan", value=default_text, height=150)
    run = st.button("Prediksi", type="primary", use_container_width=True)

    if run:
        if not text.strip():
            st.warning("Masukkan teks pesan terlebih dahulu.")
        else:
            try:
                with st.spinner("Memuat model dan menjalankan inference..."):
                    engine = get_engine(model_type, device)
                    result = engine.predict(text)
                label, level = risk_badge(result)
                st.markdown(f'<span class="badge {level}">{label}</span>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.markdown(
                    f'<div class="metric-box"><div class="metric-label">Probabilitas Scam</div>'
                    f'<div class="metric-value">{result.prob_scam:.2%}</div></div>',
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f'<div class="metric-box"><div class="metric-label">Threshold</div>'
                    f'<div class="metric-value">{result.threshold:.2f}</div></div>',
                    unsafe_allow_html=True,
                )
                c3.markdown(
                    f'<div class="metric-box"><div class="metric-label">Prediksi</div>'
                    f'<div class="metric-value">{result.prediction}</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="section-title" style="font-size:1.35rem;margin-top:18px;">Teks Setelah Preprocessing</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="clean-text">{escape(result.clean_text)}</div>', unsafe_allow_html=True)
                with st.expander("Output JSON"):
                    st.json(asdict(result))
            except DependencyMissingError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

with right:
    st.markdown(
        """
        <div class="section-title">Evaluasi Model</div>
        """,
        unsafe_allow_html=True,
    )
    if summary_rows:
        for row in summary_rows:
            with st.container(border=True):
                st.markdown(f"**{row['model']}**")
                m1, m2 = st.columns(2)
                m1.metric("F1", format_percent(row["f1"]))
                m2.metric("Accuracy", format_percent(row["accuracy"]))
                st.caption(f"Precision {format_percent(row['precision'])} | Recall {format_percent(row['recall'])}")
    else:
        st.info("Ringkasan metrik belum ditemukan.")
