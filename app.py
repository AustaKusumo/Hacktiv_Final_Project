"""Streamlit entry point for DataLens AI."""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from datalens.analysis import (
    build_dataset_context,
    create_chart,
    dataframe_profile,
    local_answer,
)
from datalens.gemini_service import ChatSettings, GeminiError, ask_gemini


load_dotenv()
ROOT = Path(__file__).parent
SAMPLE_DATA = ROOT / "sample_data" / "sales_sample.csv"

st.set_page_config(
    page_title="DataLens AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {border-right: 1px solid #e9eaf2;}
        .hero {
            padding: 2rem 2.2rem; border-radius: 24px; color: white;
            background: radial-gradient(circle at 90% 10%, #a98cff 0, transparent 34%),
                        linear-gradient(135deg, #392b78 0%, #7657ff 65%, #8f6fff 100%);
            box-shadow: 0 18px 45px rgba(82, 57, 173, .18); margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0; font-size: 2.5rem; letter-spacing: -0.04em;}
        .hero p {margin: .55rem 0 0; max-width: 700px; opacity: .9; font-size: 1.05rem;}
        .eyebrow {font-size: .76rem; text-transform: uppercase; letter-spacing: .16em; font-weight: 700; opacity: .75;}
        .metric-card {background: white; border: 1px solid #e9eaf2; border-radius: 18px; padding: 1rem 1.1rem;}
        .metric-label {color: #6d7384; font-size: .82rem;}
        .metric-value {color: #182033; font-size: 1.55rem; font-weight: 750; margin-top: .15rem;}
        .status-pill {display:inline-flex; padding:.28rem .65rem; border-radius:999px; font-size:.8rem; font-weight:650;}
        .status-live {background:#e7f8ef; color:#17764b;}
        .status-demo {background:#fff4d9; color:#8a5b00;}
        div[data-testid="stChatMessage"] {background:white; border:1px solid #ececf4; border-radius:18px; padding:.25rem .5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(content))


def load_selected_dataset(uploaded_file) -> tuple[pd.DataFrame, str, bytes]:
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        return read_csv_bytes(content), uploaded_file.name, content
    content = SAMPLE_DATA.read_bytes()
    return read_csv_bytes(content), "sales_sample.csv (contoh)", content


def reset_for_dataset(content: bytes) -> None:
    fingerprint = hashlib.sha256(content).hexdigest()
    if st.session_state.get("dataset_fingerprint") != fingerprint:
        st.session_state.dataset_fingerprint = fingerprint
        st.session_state.messages = []
        st.session_state.previous_interaction_id = None


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


inject_styles()

with st.sidebar:
    st.markdown("### ◈ DataLens AI")
    st.caption("Chatbot analisis data berbasis Gemini")
    st.divider()
    uploaded = st.file_uploader("Upload dataset", type=["csv"], help="Maksimum 25 MB")
    st.markdown("#### Konfigurasi AI")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Gemini API key",
        value=env_key,
        type="password",
        help="Disimpan hanya selama sesi dan tidak ditulis ke file.",
    )
    model = st.text_input("Model", value=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    language = st.selectbox("Bahasa", ["Bahasa Indonesia", "English"])
    style = st.selectbox("Gaya bahasa", ["Profesional", "Santai", "Akademik"])
    expertise = st.select_slider("Tingkat pengguna", ["Pemula", "Intermediate", "Expert"])
    length = st.selectbox("Panjang jawaban", ["Ringkas", "Sedang", "Terperinci"], index=1)
    temperature = st.slider("Kreativitas", 0.0, 1.0, 0.4, 0.1)
    if st.button("Hapus percakapan", width="stretch"):
        st.session_state.messages = []
        st.session_state.previous_interaction_id = None
        st.rerun()

try:
    df, dataset_name, raw_content = load_selected_dataset(uploaded)
except Exception as exc:
    st.error(f"Dataset tidak dapat dibaca: {exc}")
    st.stop()

reset_for_dataset(raw_content)
profile = dataframe_profile(df)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered data exploration</div>
      <h1>Tanyakan apa pun tentang datamu.</h1>
      <p>Unggah CSV, temukan pola, periksa kualitas data, dan ubah angka menjadi insight yang mudah dipahami.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_class = "status-live" if api_key else "status-demo"
status_text = "Gemini aktif" if api_key else "Mode demo lokal"
st.markdown(
    f'**Dataset:** `{dataset_name}` &nbsp; <span class="status-pill {status_class}">{status_text}</span>',
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
with metric_columns[0]:
    render_metric("Baris", f"{profile['rows']:,}")
with metric_columns[1]:
    render_metric("Kolom", str(profile["columns"]))
with metric_columns[2]:
    render_metric("Missing cells", f"{profile['missing_cells']:,}")
with metric_columns[3]:
    render_metric("Duplikat", f"{profile['duplicate_rows']:,}")

chat_tab, overview_tab = st.tabs(["✦ Chat Analysis", "▦ Dataset Overview"])

with chat_tab:
    st.caption("Contoh: “Buat grafik distribusi revenue” atau “Kolom mana yang memiliki missing value?”")
    if not st.session_state.messages:
        st.info(
            "Saya siap menganalisis dataset. Tanpa API key, pertanyaan umum tetap dijawab "
            "oleh analysis engine lokal."
        )

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("chart_question"):
                chart = create_chart(df, message["chart_question"])
                if chart:
                    st.plotly_chart(chart.figure, width="stretch", key=f"history-chart-{index}")
                    st.caption(chart.caption)

    question = st.chat_input("Tanyakan sesuatu tentang dataset ini…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        settings = ChatSettings(language, style, expertise, length, temperature)
        answer = ""
        used_fallback = False
        error_detail = None
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis dataset…"):
                if api_key:
                    try:
                        answer, interaction_id = ask_gemini(
                            api_key=api_key,
                            model=model,
                            question=question,
                            dataset_context=build_dataset_context(df),
                            settings=settings,
                            previous_interaction_id=st.session_state.previous_interaction_id,
                        )
                        st.session_state.previous_interaction_id = interaction_id
                    except GeminiError as exc:
                        used_fallback = True
                        error_detail = str(exc)
                        answer = local_answer(df, question)
                else:
                    used_fallback = True
                    answer = local_answer(df, question)

            st.markdown(answer)
            if error_detail:
                st.warning(f"Gemini tidak tersedia; jawaban lokal digunakan. {error_detail}")
            elif used_fallback:
                st.caption("Jawaban dibuat oleh analysis engine lokal. Tambahkan API key untuk interpretasi LLM.")
            chart = create_chart(df, question)
            if chart:
                st.plotly_chart(chart.figure, width="stretch", key=f"new-chart-{len(st.session_state.messages)}")
                st.caption(chart.caption)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "chart_question": question if chart else None}
        )

with overview_tab:
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("#### Preview data")
        st.dataframe(df.head(100), width="stretch", height=350)
    with right:
        st.markdown("#### Kualitas data")
        quality = pd.DataFrame(
            {
                "Kolom": df.columns.astype(str),
                "Tipe": [str(dtype) for dtype in df.dtypes],
                "Missing": df.isna().sum().values,
                "Unik": df.nunique(dropna=True).values,
            }
        )
        st.dataframe(quality, width="stretch", hide_index=True, height=350)

    st.markdown("#### Statistik numerik")
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        st.info("Dataset tidak memiliki kolom numerik.")
    else:
        st.dataframe(numeric.describe().T, width="stretch")

st.caption("DataLens AI · Final Project LLM-Based Tools & Gemini API Integration for Data Scientists")
