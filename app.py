import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import base64

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taksir Rumah",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD IMAGES AS BASE64 ────────────────────────────────────────────────────
def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = os.path.dirname(__file__)
bg1_b64 = img_to_b64(os.path.join(BASE_DIR, "gambar", "background 1.jpg"))
bg2_b64 = img_to_b64(os.path.join(BASE_DIR, "gambar", "background 2.jpg"))

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    base = os.path.join(BASE_DIR, "model_rf_rumah")
    encoder = joblib.load(os.path.join(base, "encoder_lokasi.pkl"))
    model   = joblib.load(os.path.join(base, "model_rf_rumah.pkl"))
    return encoder, model

encoder, model = load_model()
LOKASI_LIST = list(encoder.classes_)

# ─── FORMAT HARGA ─────────────────────────────────────────────────────────────
def format_harga(nilai: float) -> str:
    if nilai >= 1_000:
        m = nilai / 1_000
        return f"Rp {m:,.2f} Miliar" if abs(m - round(m)) >= 0.005 else f"Rp {int(round(m)):,} Miliar"
    return f"Rp {nilai:,.1f} Juta" if abs(nilai - round(nilai)) >= 0.05 else f"Rp {int(round(nilai)):,} Juta"

# ─── GLOBAL BASE CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #0a0a0a;
    color: #e0e0e0;
}

/* hide streamlit chrome */
#MainMenu, footer,
[data-testid="stDecoration"],
[data-testid="stElementToolbar"] { display: none !important; }
[data-testid="stToolbar"] { right: 2rem !important; }

/* make header visible so hamburger shows */
header, [data-testid="stHeader"] { 
    background: #0a0a0a !important; 
    z-index: 999999 !important; 
}

/* main block */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ════════════════════════════════
   SIDEBAR
   ════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #1a1a1a !important;
}

/* All sidebar text inherits Inter */
[data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif !important;
    color: #555 !important;
}
/* Restore icon fonts just in case */
[data-testid="stSidebar"] .material-symbols-rounded {
    font-family: "Material Symbols Rounded" !important;
}

/* ── Hide the "Menu" label above radio ── */
[data-testid="stSidebar"] .stRadio > div:first-child {
    display: none !important;
}

/* ── Radio group: remove gap ── */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 0 !important;
    flex-direction: column !important;
}

/* ── Individual radio row ── */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    gap: 0 !important;
    padding: 12px 24px !important;
    border-radius: 0 !important;
    margin: 0 !important;
    font-size: .82rem !important;
    font-weight: 400 !important;
    color: #555 !important;
    letter-spacing: .3px !important;
    cursor: pointer !important;
    border-left: 2px solid transparent !important;
    background: transparent !important;
    transition: color .15s, background .15s !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    color: #bbb !important;
    background: #141414 !important;
}

/* ── Hide the circular radio dot (the Streamlit default indicator) ── */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* ── The text span inside the label ── */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:last-child,
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:last-child p {
    font-size: .82rem !important;
    color: inherit !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ── Selected state: override via aria-checked on parent ── */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
    color: #e0e0e0 !important;
    border-left: 2px solid #3a7bd5 !important;
    background: #141414 !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) > div:last-child,
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) > div:last-child p {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}

/* ── Hamburger / collapse button ── */
[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="stSidebarCollapsedControl"] {
    color: #ffffff !important;
    background-color: transparent !important;
}
[data-testid="collapsedControl"] svg,
button[kind="header"] svg,
[data-testid="stHeader"] button svg,
[data-testid="stToolbar"] button svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
}

/* ── Sidebar toggle arrow (expand when collapsed) ── */
.st-emotion-cache-pb6fr7,
[data-testid="stSidebarNavCollapseIcon"] {
    color: #555 !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 2px; }

/* removed custom media queries to let Streamlit handle mobile responsiveness natively */
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="padding: 32px 24px 24px;">
        <div style="
            font-size: .58rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #a0a0a0;
            margin-bottom: 8px;
        ">Sistem Prediksi</div>
        <div style="
            font-size: 1rem;
            font-weight: 700;
            color: #e0e0e0;
            line-height: 1.3;
            letter-spacing: -.2px;
        ">Taksir Rumah</div>
        <div style="width:20px; height:2px; background:#3a7bd5; margin-top:10px;"></div>
    </div>
    <div style="height:1px; background:#1a1a1a; margin: 0 24px 16px;"></div>
    <div style="
        font-size: .55rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #a0a0a0;
        padding: 0 24px 10px;
    ">Menu</div>
    """, unsafe_allow_html=True)

    halaman = st.radio(
        "Menu",
        options=["Beranda", "Prediksi Harga"],
        label_visibility="collapsed",
    )

    # Footer — NOT absolute, flows naturally below radio
    st.markdown("""
    <div style="
        margin-top: 40px;
        padding: 16px 24px;
        border-top: 1px solid #1a1a1a;
    ">
        <div style="font-size:.62rem; color:#999999; line-height:1.9;">
            Random Forest Model<br>
            Data Properti Yogyakarta
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if halaman == "Beranda":

    # ── Hero (background image + heavy overlay) ──────────────────────────────
    st.markdown(f"""
    <style>
    /* Reset page background to dark */
    [data-testid="stAppViewContainer"] {{ background: #0a0a0a !important; }}
    .main .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}
    </style>

    <div style="
        position: relative;
        width: 100%;
        height: 380px;
        overflow: hidden;
        display: flex;
        align-items: flex-end;
        padding: 48px 64px;
    ">
        <!-- background image -->
        <div style="
            position: absolute; inset: 0;
            background-image: url('data:image/jpeg;base64,{bg1_b64}');
            background-size: cover;
            background-position: center 40%;
            filter: brightness(0.65) saturate(0.8);
        "></div>
        <!-- gradient overlay bottom fade -->
        <div style="
            position: absolute; inset: 0;
            background: linear-gradient(to bottom,
                rgba(10,10,10,0.2) 0%,
                rgba(10,10,10,0.8) 75%,
                rgba(10,10,10,1) 100%
            );
        "></div>
        <!-- content -->
        <div style="position: relative; z-index: 1;">
            <div style="
                font-size: .6rem; letter-spacing: 3px; text-transform: uppercase;
                color: #3a7bd5; margin-bottom: 12px;
            ">Sistem Prediksi Properti</div>
            <h1 style="
                font-size: 2rem; font-weight: 700; color: #fff;
                letter-spacing: -.5px; line-height: 1.2; margin: 0 0 10px;
            ">Taksir Rumah &mdash; Perhitungan Pintar</h1>
            <p style="
                font-size: .85rem; color: #999; max-width: 520px;
                line-height: 1.7; margin: 0;
            ">
                Prediksi harga rumah berbasis machine learning untuk wilayah
                Daerah Istimewa Yogyakarta. Masukkan spesifikasi properti,
                sistem akan memperkirakan nilai pasarnya secara instan.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Body content (solid dark background) ─────────────────────────────────
    st.markdown("""
    <div style="background:#0a0a0a; padding: 0 64px 64px;">
    """, unsafe_allow_html=True)

    # Stat row
    stats = [("70+", "Lokasi Tercakup"), ("Random Forest", "Algoritma"), ("8", "Fitur Input"), ("DIY", "Wilayah")]
    cols = st.columns(len(stats), gap="small")
    for col, (num, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div style="
                border: 1px solid #1c1c1c;
                border-top: 1px solid #262626;
                border-radius: 4px;
                padding: 20px 20px 16px;
                background: #111;
            ">
                <div style="font-size:1.2rem; font-weight:700; color:#fff;
                            letter-spacing:-.3px; margin-bottom:4px;">{num}</div>
                <div style="font-size:.72rem; color:#444;
                            text-transform:uppercase; letter-spacing:1px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Section: Tentang
    st.markdown("""
    <div style="margin-top:48px; margin-bottom:20px;">
        <div style="font-size:.6rem; letter-spacing:3px; text-transform:uppercase;
                    color:#3a7bd5; margin-bottom:16px;">Tentang Sistem</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    card_css = """
        background:#111; border:1px solid #1c1c1c;
        border-radius:4px; padding:24px 20px; height:100%;
    """
    for col, title, desc in zip(
        [c1, c2, c3],
        ["Algoritma Random Forest",
         "Cakupan 70+ Lokasi",
         "Hasil Instan"],
        [
            "Model ensemble decision tree yang dilatih pada data listing properti nyata di Yogyakarta.",
            "Mencakup kecamatan di Sleman, Bantul, Kota Yogyakarta, Kulon Progo, dan Gunung Kidul.",
            "Estimasi harga muncul dalam hitungan detik setelah spesifikasi properti dimasukkan.",
        ],
    ):
        with col:
            st.markdown(f"""
            <div style="{card_css}">
                <div style="width:20px; height:2px; background:#3a7bd5; margin-bottom:16px;"></div>
                <div style="font-size:.88rem; font-weight:600; color:#e0e0e0;
                            margin-bottom:8px;">{title}</div>
                <div style="font-size:.78rem; color:#555; line-height:1.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Section: Cara Kerja
    st.markdown("""
    <div style="margin-top:48px; margin-bottom:20px;">
        <div style="font-size:.6rem; letter-spacing:3px; text-transform:uppercase;
                    color:#3a7bd5; margin-bottom:0;">Cara Kerja</div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Masukkan Data Properti",
         "Isi formulir dengan lokasi, jumlah kamar, luas tanah, luas bangunan, dan kapasitas garasi."),
        ("02", "Preprocessing Otomatis",
         "Sistem mengenkode lokasi dan menghitung fitur turunan: rasio bangunan dan total kamar."),
        ("03", "Prediksi oleh Model",
         "Data diproses oleh Random Forest untuk menghasilkan estimasi harga pasar."),
        ("04", "Lihat Hasil",
         "Harga estimasi ditampilkan dalam format Rupiah yang jelas dan mudah dipahami."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="
            display: flex; gap: 24px; align-items: flex-start;
            padding: 16px 0; border-bottom: 1px solid #141414;
        ">
            <div style="
                font-size: .65rem; font-weight: 700; color: #3a7bd5;
                min-width: 24px; padding-top: 2px; letter-spacing: .5px;
            ">{num}</div>
            <div>
                <div style="font-size:.85rem; font-weight:600; color:#ccc;
                            margin-bottom:4px;">{title}</div>
                <div style="font-size:.78rem; color:#aaaaaa; line-height:1.7;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Section: Tabel Fitur
    st.markdown("""
    <div style="margin-top:48px; margin-bottom:16px;">
        <div style="font-size:.6rem; letter-spacing:3px; text-transform:uppercase;
                    color:#3a7bd5;">Fitur Model</div>
    </div>
    """, unsafe_allow_html=True)

    fitur_df = pd.DataFrame({
        "Fitur": ["Lokasi","Kamar Tidur","Kamar Mandi","Garasi",
                  "Luas Tanah (m2)","Luas Bangunan (m2)","Rasio Bangunan","Total Kamar"],
        "Keterangan": [
            "Kecamatan / kabupaten properti (di-encode otomatis)",
            "Jumlah kamar tidur",
            "Jumlah kamar mandi",
            "Kapasitas garasi dalam jumlah mobil",
            "Luas tanah dalam meter persegi",
            "Luas bangunan dalam meter persegi",
            "Luas Bangunan dibagi Luas Tanah — otomatis",
            "Kamar Tidur ditambah Kamar Mandi — otomatis",
        ],
        "Sumber": ["Manual","Manual","Manual","Manual",
                   "Manual","Manual","Otomatis","Otomatis"],
    })
    st.dataframe(
        fitur_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Fitur":      st.column_config.TextColumn("Fitur",      width="medium"),
            "Keterangan": st.column_config.TextColumn("Keterangan"),
            "Sumber":     st.column_config.TextColumn("Sumber",     width="small"),
        },
    )

    st.markdown("""
    <div style="margin-top:32px; padding:14px 18px;
                border:1px solid #1c1c1c; border-left:2px solid #3a7bd5;
                border-radius:2px; background:#111;">
        <span style="font-size:.78rem; color:#555;">
            Pilih <strong style="color:#888;">Prediksi Harga</strong>
            di sidebar kiri untuk mulai menggunakan sistem.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
else:

    # ── Hero strip ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{ 
        background-image: linear-gradient(rgba(10, 10, 10, 0.82), rgba(10, 10, 10, 0.95)), url('data:image/jpeg;base64,{bg2_b64}') !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        background-position: center !important;
    }}
    .main .block-container {{ padding: 0 !important; max-width: 100% !important; }}

    /* input style */
    [data-testid="stSelectbox"] > div > div {{
        background: #111 !important;
        border: 1px solid #1c1c1c !important;
        border-radius: 4px !important;
        color: #e0e0e0 !important;
        font-size: .83rem !important;
    }}
    [data-testid="stSelectbox"] > div > div:hover {{
        border-color: #2a2a2a !important;
    }}
    [data-testid="stSelectbox"] svg {{ fill: #555 !important; }}
    [data-testid="stNumberInput"] input {{
        background: #111 !important;
        border: 1px solid #1c1c1c !important;
        border-radius: 4px !important;
        color: #e0e0e0 !important;
        font-size: .83rem !important;
    }}
    [data-testid="stNumberInput"] input:focus {{
        border-color: #3a7bd5 !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    [data-testid="stNumberInput"] button {{
        background: #161616 !important;
        border-color: #1c1c1c !important;
        color: #555 !important;
    }}
    /* labels */
    [data-testid="stWidgetLabel"] p,
    .stSelectbox label, .stNumberInput label {{
        color: #444 !important;
        font-size: .7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }}
    /* button */
    .stButton > button {{
        background: #161f2e !important;
        color: #7aabf0 !important;
        border: 1px solid #1e3458 !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .8rem !important;
        font-weight: 500 !important;
        letter-spacing: .8px !important;
        text-transform: uppercase !important;
        width: 100% !important;
        transition: all .15s !important;
    }}
    .stButton > button:hover {{
        background: #1a2a47 !important;
        color: #a0c4f8 !important;
        border-color: #2a4a78 !important;
    }}
    </style>

    <div style="
        position: relative;
        width: 100%; height: 180px;
        display: flex; align-items: flex-end;
        padding: 36px 64px;
    ">
        <div style="position:relative; z-index:1;">
            <div style="font-size:.6rem; letter-spacing:3px; text-transform:uppercase;
                        color:#3a7bd5; margin-bottom:10px;">Prediksi Harga</div>
            <h1 style="font-size:1.6rem; font-weight:700; color:#fff;
                       letter-spacing:-.3px; margin:0;">
                Estimasi Harga Rumah
            </h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Body ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:transparent; padding: 8px 64px 64px;">
    <p style="font-size:.8rem; color:#b3b3b3; margin-bottom:36px;">
        Isi spesifikasi properti untuk mendapatkan estimasi harga pasar.
    </p>
    </div>
    """, unsafe_allow_html=True)

    pad_l, col_form, pad_mid, col_result, pad_r = st.columns([1, 4, 1, 4, 1])

    with col_form:

        # Group: Lokasi
        st.markdown("""
        <div style="font-size:.6rem; letter-spacing:2.5px; text-transform:uppercase;
                    color:#a0a0a0; margin-bottom:12px; margin-top:8px;">Lokasi</div>
        """, unsafe_allow_html=True)

        lokasi_input = st.selectbox(
            "Masukkan Lokasi (Contoh: Ngaglik, Sleman)",
            options=LOKASI_LIST,
            index=LOKASI_LIST.index("Ngaglik, Sleman") if "Ngaglik, Sleman" in LOKASI_LIST else 0,
            key="lokasi",
        )

        # Group: Kamar
        st.markdown("""
        <div style="height:20px;"></div>
        <div style="font-size:.6rem; letter-spacing:2.5px; text-transform:uppercase;
                    color:#a0a0a0; margin-bottom:12px;">Kamar & Garasi</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            kamar_tidur = st.number_input("Masukkan Jumlah Kamar Tidur",
                                          min_value=1, max_value=20, value=2, step=1, key="ktidur")
        with c2:
            kamar_mandi = st.number_input("Masukkan Jumlah Kamar Mandi",
                                          min_value=1, max_value=10, value=1, step=1, key="kmandi")

        garasi = st.number_input("Masukkan Kapasitas Garasi (Mobil)",
                                 min_value=0, max_value=10, value=1, step=1, key="garasi")

        # Group: Luas
        st.markdown("""
        <div style="height:20px;"></div>
        <div style="font-size:.6rem; letter-spacing:2.5px; text-transform:uppercase;
                    color:#a0a0a0; margin-bottom:12px;">Luas</div>
        """, unsafe_allow_html=True)

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            luas_tanah = st.number_input("Masukkan Luas Tanah (m2)",
                                         min_value=10, max_value=10000, value=120, step=1, key="tanah")
        with c4:
            luas_bangunan = st.number_input("Masukkan Luas Bangunan (m2)",
                                            min_value=10, max_value=10000, value=100, step=1, key="bangunan")

        rasio       = round(luas_bangunan / luas_tanah, 4) if luas_tanah > 0 else 0
        total_kamar = kamar_tidur + kamar_mandi

        st.markdown(f"""
        <div style="
            margin: 16px 0;
            padding: 10px 14px;
            border: 1px solid #161616;
            border-left: 2px solid #1e3458;
            border-radius: 2px;
            background: #0d0d0d;
            font-size: .72rem;
            color: #b3b3b3;
            display: flex; gap: 20px;
        ">
            <span>Rasio Bangunan: <strong style="color:#3a5a8a;">{rasio:.2%}</strong></span>
            <span>Total Kamar: <strong style="color:#3a5a8a;">{total_kamar}</strong></span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Prediksi Harga", key="predict")

    # ── RESULT COLUMN ─────────────────────────────────────────────────────────
    with col_result:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        if predict_btn:
            with st.spinner(""):
                try:
                    lokasi_encoded = encoder.transform([lokasi_input])[0]
                    input_df = pd.DataFrame([[
                        lokasi_encoded, kamar_tidur, kamar_mandi, garasi,
                        luas_tanah, luas_bangunan, rasio, total_kamar,
                    ]], columns=[
                        "Lokasi","Kamar_Tidur","Kamar_Mandi","Garasi",
                        "Luas_Tanah_m2","Luas_Bangunan_m2","Rasio_Bangunan","Total_Kamar",
                    ])

                    harga_pred = model.predict(input_df)[0]
                    harga_str  = format_harga(harga_pred)

                    # Result card
                    st.markdown(f"""
                    <div style="
                        border: 1px solid #1c1c1c;
                        border-top: 2px solid #3a7bd5;
                        border-radius: 4px;
                        background: #0f0f0f;
                        padding: 28px 24px;
                        margin-bottom: 16px;
                    ">
                        <div style="
                            font-size:.6rem; letter-spacing:3px;
                            text-transform:uppercase; color:#3a7bd5;
                            margin-bottom:12px;
                        ">Estimasi Harga Pasar</div>
                        <div style="
                            font-size:2rem; font-weight:700;
                            color:#ffffff; letter-spacing:-.5px;
                            line-height:1.15; margin-bottom:6px;
                        ">{harga_str}</div>
                        <div style="font-size:.75rem; color:#b3b3b3;">
                            {harga_pred:,.0f} Juta Rupiah
                        </div>
                        <div style="
                            margin-top:20px; padding-top:16px;
                            border-top:1px solid #161616;
                            font-size:.72rem; color:#999999; line-height:1.7;
                        ">
                            Estimasi bersifat indikatif berdasarkan model machine learning.
                            Nilai aktual dapat berbeda tergantung kondisi pasar.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detail input
                    items = [
                        ("Lokasi",        lokasi_input),
                        ("Kamar Tidur",   str(kamar_tidur)),
                        ("Kamar Mandi",   str(kamar_mandi)),
                        ("Garasi",        str(garasi)),
                        ("Luas Tanah",    f"{luas_tanah} m2"),
                        ("Luas Bangunan", f"{luas_bangunan} m2"),
                    ]
                    pairs = [items[i:i+2] for i in range(0, len(items), 2)]
                    rows_html = ""
                    for pair in pairs:
                        rows_html += "<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;'>"
                        for label, val in pair:
                            rows_html += f"""
                            <div style="background:#0d0d0d;border:1px solid #161616;
                                        border-radius:4px;padding:12px 14px;">
                                <div style="font-size:.6rem;text-transform:uppercase;
                                            letter-spacing:1px;color:#a0a0a0;margin-bottom:4px;">{label}</div>
                                <div style="font-size:.82rem;font-weight:600;color:#8aabdc;">{val}</div>
                            </div>"""
                        rows_html += "</div>"

                    st.markdown(f"""
                    <div style="font-size:.6rem;letter-spacing:2.5px;text-transform:uppercase;
                                color:#a0a0a0;margin-bottom:10px;">Detail Input</div>
                    {rows_html}
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")

        else:
            st.markdown("""
            <div style="
                border: 1px solid #141414;
                border-radius: 4px;
                background: #0d0d0d;
                padding: 48px 24px;
                text-align: center;
            ">
                <div style="width:32px; height:2px; background:#1c1c1c;
                            margin: 0 auto 20px;"></div>
                <div style="font-size:.72rem; color:#a0a0a0; line-height:1.8;">
                    Isi formulir dan klik<br>
                    <span style="color:#3a5a8a; font-weight:500;">Prediksi Harga</span><br>
                    untuk melihat estimasi.
                </div>
            </div>

            <div style="margin-top:24px;">
                <div style="font-size:.6rem;letter-spacing:2.5px;text-transform:uppercase;
                            color:#a0a0a0;margin-bottom:12px;">Panduan</div>
            </div>
            """, unsafe_allow_html=True)

            panduan = [
                ("Lokasi",   "Pilih kecamatan dan kabupaten yang paling sesuai."),
                ("Luas",     "Masukkan luas dalam satuan meter persegi (m2)."),
                ("Garasi",   "Isi 0 jika tidak ada garasi."),
                ("Hasil",    "Estimasi merupakan nilai pasar, bukan harga jual resmi."),
            ]
            for label, desc in panduan:
                st.markdown(f"""
                <div style="
                    display:flex; gap:16px;
                    padding:10px 0; border-bottom:1px solid #111;
                    align-items:flex-start;
                ">
                    <div style="font-size:.62rem;font-weight:600;color:#1e3458;
                                min-width:56px;text-transform:uppercase;
                                letter-spacing:.5px;padding-top:1px;">{label}</div>
                    <div style="font-size:.76rem;color:#b3b3b3;line-height:1.7;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
