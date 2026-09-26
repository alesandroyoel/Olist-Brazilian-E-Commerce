import os
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# folium bersifat opsional: dashboard tetap berjalan (fallback ke chart matplotlib)
# meskipun folium/streamlit-folium belum terpasang atau tidak ada koneksi internet
# untuk mengambil tile peta.
try:
    import folium
    import branca.colormap as bcm
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Titik tengah (centroid) tiap state, dirata-rata dari geolocation_dataset.csv
# (disimpan statis di sini agar dashboard tidak perlu memuat file geolocation
# yang berukuran puluhan MB hanya untuk mengambil koordinat).
STATE_COORDS = {
    "AC": (-9.7093, -68.4533), "AL": (-9.5924, -36.0676), "AM": (-3.3466, -60.5357),
    "AP": (0.0738, -51.2045), "BA": (-13.0624, -39.6272), "CE": (-4.3952, -39.0283),
    "DF": (-15.8143, -47.9797), "ES": (-20.0781, -40.5161), "GO": (-16.5674, -49.3444),
    "MA": (-3.8212, -44.8183), "MG": (-19.8574, -44.4494), "MS": (-20.7908, -54.5352),
    "MT": (-14.2141, -55.7154), "PA": (-2.6680, -49.5292), "PB": (-7.0843, -35.8582),
    "PE": (-8.1882, -35.7870), "PI": (-5.7002, -42.4870), "PR": (-24.7617, -50.9507),
    "RJ": (-22.7085, -43.1503), "RN": (-5.8512, -36.0101), "RO": (-10.3584, -62.7077),
    "RR": (2.7288, -60.6783), "RS": (-29.6497, -52.0649), "SC": (-27.2215, -49.6358),
    "SE": (-10.8498, -37.2002), "SP": (-23.0867, -47.1618), "TO": (-9.4912, -48.3577),
}

# Nama lengkap tiap state (untuk mencocokkan properti "name" pada GeoJSON batas wilayah)
ABBR_TO_NAME = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí",
    "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul",
    "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
    "SE": "Sergipe", "TO": "Tocantins",
}

# Tile basemap: memakai Esri World Street Map (gratis, tanpa API key, dan stabil).
# CartoDB positron sekarang mewajibkan API key, sehingga tidak dipakai lagi.
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, "
    "Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012"
)

BRAZIL_GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/brazil-states.geojson"
)


def _normalize_text(s):
    """Hilangkan aksen & ubah ke huruf kecil, untuk mencocokkan nama provinsi yang mungkin
    ditulis beda format antara dataset kita dengan GeoJSON pihak ketiga."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s.strip().lower()


_NAME_TO_ABBR_NORM = {_normalize_text(v): k for k, v in ABBR_TO_NAME.items()}


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_brazil_geojson():
    """Ambil GeoJSON batas wilayah state Brasil, lalu suntik properti `_state_code`
    (kode 2 huruf) ke tiap fitur berdasarkan pencocokan nama (anti sensitif format/aksen).
    Mengembalikan None jika gagal diambil atau jumlah state yang cocok terlalu sedikit,
    sehingga dashboard bisa otomatis fallback ke mode bubble map biasa."""
    if not REQUESTS_AVAILABLE:
        return None
    try:
        resp = requests.get(BRAZIL_GEOJSON_URL, timeout=10)
        resp.raise_for_status()
        gj = resp.json()
    except Exception:
        return None

    matched = 0
    for feature in gj.get("features", []):
        props = feature.get("properties") or {}
        code = None
        for val in props.values():
            if isinstance(val, str) and val.strip().upper() in ABBR_TO_NAME:
                code = val.strip().upper()
                break
        if code is None:
            for val in props.values():
                norm = _normalize_text(val)
                if norm and norm in _NAME_TO_ABBR_NORM:
                    code = _NAME_TO_ABBR_NORM[norm]
                    break
        # properti `_state_code` WAJIB ada di semua fitur (folium.Choropleth butuh key_on
        # yang konsisten); fitur yang tidak cocok diberi kode sentinel "XX" (dianggap
        # data kosong / nan_fill_color) alih-alih dibiarkan tanpa properti tersebut.
        feature.setdefault("properties", {})["_state_code"] = code or "XX"
        if code:
            matched += 1

    if matched < 20:  # dari 27 state; kalau sebagian besar gagal cocok, jangan dipakai
        return None
    return gj

# ----------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2E86AB"
SECONDARY = "#F18F01"
DANGER = "#C1272D"
GOOD = "#2E7D32"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["font.size"] = 10

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    h1, h2, h3 {color: #14213D;}
    /* Gaya navigasi ala-tab untuk widget radio (menggantikan st.tabs agar
       tidak reset ke tab pertama setiap kali filter di sidebar diubah) */
    div[role="radiogroup"] {
        flex-direction: row; gap: 0.4rem; flex-wrap: wrap; border-bottom: 2px solid #E6E6E6;
        padding-bottom: 0;
    }
    div[role="radiogroup"] label {
        background: #333333; padding: 0.55rem 1rem; border-radius: 8px 8px 0 0;
        border: 1px solid #E0E0E0; border-bottom: none; margin-bottom: -2px; font-weight: 600;
    }
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "main_data.csv")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "order_month" not in df.columns:
        df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df


try:
    main_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        "File `main_data.csv` tidak ditemukan di folder `dashboard/`. "
        "Pastikan file tersebut berada satu folder dengan `dashboard.py`."
    )
    st.stop()

# ----------------------------------------------------------------------
# SIDEBAR - FILTER
# ----------------------------------------------------------------------
st.sidebar.title("🛍️ Brazilian E-Commerce")
st.sidebar.caption("Olist Brazilian E-Commerce Public Dataset")
st.sidebar.markdown("---")
st.sidebar.header("Filter Data")

min_date = main_df["order_purchase_timestamp"].min().date()
max_date = main_df["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Rentang tanggal order",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

state_options = sorted(main_df["customer_state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect(
    "Provinsi/State pelanggan", options=state_options, default=[]
)

cat_options = sorted(main_df["main_category"].dropna().unique().tolist())
selected_cats = st.sidebar.multiselect(
    "Kategori produk", options=cat_options, default=[]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Oleh Alesandro Yoel Deca Putranto • [GitHub](https://github.com/alesandrodecaputranto)"
)

# apply filters
mask = (
    (main_df["order_purchase_timestamp"].dt.date >= start_date)
    & (main_df["order_purchase_timestamp"].dt.date <= end_date)
)
if selected_states:
    mask &= main_df["customer_state"].isin(selected_states)
if selected_cats:
    mask &= main_df["main_category"].isin(selected_cats)

df = main_df.loc[mask].copy()

if df.empty:
    st.warning("Tidak ada data pada kombinasi filter yang dipilih. Silakan ubah filter.")
    st.stop()

# ----------------------------------------------------------------------
# HEADER & KPI
# ----------------------------------------------------------------------
st.title("🛍️ Olist E-Commerce Dashboard")
st.markdown(
    "Dashboard interaktif untuk menganalisis **tren penjualan**, "
    "**pengaruh keterlambatan pengiriman terhadap kepuasan pelanggan**, "
    "serta **segmentasi pelanggan (RFM)** dan **distribusi geografis (Geospatial)** "
    "pada *E-Commerce Public Dataset (Olist Brazilian E-Commerce)*."
)

total_revenue = df["order_value"].sum()
total_orders = df["order_id"].nunique()
avg_review = df["review_score"].mean()
pct_late = (df["delivery_delay_days"] > 0).mean() * 100 if df["delivery_delay_days"].notna().any() else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
col2.metric("Total Order", f"{total_orders:,}")
col3.metric("Rata-rata Review Score", f"{avg_review:.2f} / 5.0" if pd.notna(avg_review) else "N/A")
col4.metric("% Order Terlambat", f"{pct_late:.1f}%" if pd.notna(pct_late) else "N/A")

st.markdown("---")

# ----------------------------------------------------------------------
# NAVIGASI (radio ber-state, TIDAK reset ke menu awal saat filter diubah)
# ----------------------------------------------------------------------
TAB_LABELS = [
    "📈 Tren & Kategori Produk",
    "🚚 Keterlambatan vs Review",
    "👥 RFM Segmentation",
    "🗺️ Geospatial Analysis",
]
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = TAB_LABELS[0]

active_tab = st.radio(
    "Navigasi dashboard", TAB_LABELS, key="active_tab", horizontal=True, label_visibility="collapsed"
)

# ======================================================================
# TAB 1: Pertanyaan Bisnis 1
# ======================================================================
if active_tab == TAB_LABELS[0]:
    st.subheader("Pertanyaan Bisnis 1")
    st.markdown(
        "> Bagaimana tren jumlah pesanan dan revenue bulanan, serta kategori produk apa "
        "saja yang menjadi kontributor revenue terbesar?"
    )

    monthly = (
        df.groupby("order_month")
        .agg(n_orders=("order_id", "nunique"), revenue=("order_value", "sum"))
        .reset_index()
        .sort_values("order_month")
    )

    fig1, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(monthly["order_month"], monthly["n_orders"], marker="o", color=PRIMARY, label="Jumlah Order")
    ax1.set_xlabel("Bulan")
    ax1.set_ylabel("Jumlah Order", color=PRIMARY)
    ax1.tick_params(axis="y", labelcolor=PRIMARY)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(monthly["order_month"], monthly["revenue"], marker="s", color=SECONDARY, label="Revenue (R$)")
    ax2.set_ylabel("Revenue (R$)", color=SECONDARY)
    ax2.tick_params(axis="y", labelcolor=SECONDARY)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}jt"))

    ax1.set_title("Tren Bulanan Jumlah Order dan Revenue", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("##### 10 Kategori Produk dengan Revenue Tertinggi")
    n_cat = st.slider("Jumlah kategori yang ditampilkan", min_value=5, max_value=20, value=10, key="n_cat")
    cat_rev = (
        df.groupby("main_category")
        .agg(revenue=("order_value", "sum"), n_orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(n_cat)
    )
    cat_rev["revenue_jt"] = cat_rev["revenue"] / 1e6

    fig2, ax = plt.subplots(figsize=(10, max(3.5, n_cat * 0.4)))
    ax.barh(cat_rev["main_category"][::-1], cat_rev["revenue_jt"][::-1], color=sns.color_palette("viridis", n_cat))
    ax.set_xlabel("Revenue (Juta R$)")
    ax.set_title(f"{n_cat} Kategori Produk dengan Revenue Tertinggi", fontsize=13, fontweight="bold")
    for i, v in enumerate(cat_rev["revenue_jt"][::-1]):
        ax.text(v + max(cat_rev["revenue_jt"]) * 0.01, i, f"{v:.2f}jt", va="center", fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("Lihat data tabel kategori"):
        st.dataframe(cat_rev[["main_category", "revenue", "n_orders"]], use_container_width=True)

# ======================================================================
# TAB 2: Pertanyaan Bisnis 2
# ======================================================================
if active_tab == TAB_LABELS[1]:
    st.subheader("Pertanyaan Bisnis 2")
    st.markdown(
        "> Seberapa besar pengaruh keterlambatan pengiriman (*delivery delay*) terhadap "
        "penurunan review score pelanggan?"
    )

    df_review = df.dropna(subset=["review_score", "delivery_delay_days"]).copy()

    if df_review.empty:
        st.info("Tidak ada data review yang lengkap pada filter saat ini.")
    else:
        df_review["delivery_status"] = np.where(
            df_review["delivery_delay_days"] > 0, "Terlambat", "Tepat/Lebih Cepat"
        )
        corr = df_review["delivery_delay_days"].corr(df_review["review_score"])

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Rata-rata Review (Tepat Waktu)",
            f"{df_review.loc[df_review['delivery_status']=='Tepat/Lebih Cepat', 'review_score'].mean():.2f}",
        )
        c2.metric(
            "Rata-rata Review (Terlambat)",
            f"{df_review.loc[df_review['delivery_status']=='Terlambat', 'review_score'].mean():.2f}",
        )
        c3.metric("Korelasi Delay vs Review Score", f"{corr:.3f}")

        bins = [-1000, -1, 0, 3, 7, 15, 1000]
        labels = ["Lebih cepat", "Tepat waktu", "Telat 1-3 hari", "Telat 4-7 hari", "Telat 8-15 hari", "Telat >15 hari"]
        df_review["delay_bin"] = pd.cut(df_review["delivery_delay_days"], bins=bins, labels=labels)
        delay_score = df_review.groupby("delay_bin", observed=True)["review_score"].agg(["mean", "count"]).reindex(labels)

        colL, colR = st.columns(2)
        with colL:
            fig3, ax = plt.subplots(figsize=(6.5, 5))
            colors = [GOOD, "#66BB6A", "#FDD835", "#FB8C00", "#E64A19", DANGER]
            ax.bar(delay_score.index, delay_score["mean"], color=colors)
            ax.set_ylabel("Rata-rata Review Score")
            ax.set_title("Review Score vs Ketepatan Pengiriman", fontsize=12, fontweight="bold")
            ax.set_xticklabels(delay_score.index, rotation=25, ha="right")
            ax.set_ylim(0, 5.5)
            for i, v in enumerate(delay_score["mean"]):
                if pd.notna(v):
                    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        with colR:
            fig4, ax = plt.subplots(figsize=(6.5, 5))
            sns.countplot(
                data=df_review, x="review_score", hue="delivery_status",
                palette={"Tepat/Lebih Cepat": PRIMARY, "Terlambat": DANGER}, ax=ax
            )
            ax.set_title("Distribusi Review Score", fontsize=12, fontweight="bold")
            ax.set_xlabel("Review Score")
            ax.set_ylabel("Jumlah Order")
            ax.legend(title="Status Pengiriman")
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

        pct_1star_late = (df_review.loc[df_review["delivery_delay_days"] > 0, "review_score"] == 1).mean() * 100
        pct_1star_ontime = (df_review.loc[df_review["delivery_delay_days"] <= 0, "review_score"] == 1).mean() * 100
        st.info(
            f"📌 **{pct_1star_late:.1f}%** order yang terlambat berujung review **1 bintang**, "
            f"dibanding hanya **{pct_1star_ontime:.1f}%** pada order yang tepat waktu/lebih cepat."
        )

# ======================================================================
# TAB 3: RFM
# ======================================================================
if active_tab == TAB_LABELS[2]:
    st.subheader("Analisis Lanjutan: RFM Segmentation")
    st.markdown(
        "Segmentasi pelanggan berdasarkan **Recency** (terakhir kali belanja), "
        "**Frequency** (jumlah transaksi), dan **Monetary** (total pengeluaran)."
    )

    rfm_base = df.dropna(subset=["customer_unique_id"]).copy()
    snapshot_date = rfm_base["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = rfm_base.groupby("customer_unique_id").agg(
        Recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("order_value", "sum"),
    ).reset_index()

    if len(rfm) < 8:
        st.info("Jumlah pelanggan pada filter saat ini terlalu sedikit untuk segmentasi RFM (butuh minimal beberapa puluh pelanggan).")
    else:
        rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
        rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

        def segment(row):
            if row["RFM_score"] >= 10:
                return "Champions/Loyal"
            elif row["RFM_score"] >= 7:
                return "Potential Loyalist"
            elif row["RFM_score"] >= 5:
                return "At Risk"
            else:
                return "Hibernating"

        rfm["Segment"] = rfm.apply(segment, axis=1)

        order_seg = ["Champions/Loyal", "Potential Loyalist", "At Risk", "Hibernating"]
        seg_counts = rfm["Segment"].value_counts().reindex(order_seg).fillna(0)

        colL, colR = st.columns([1, 1.2])
        with colL:
            fig5, ax = plt.subplots(figsize=(6, 6))
            colors5 = [GOOD, PRIMARY, "#F9A825", DANGER]
            valid = seg_counts[seg_counts > 0]
            ax.pie(
                valid, labels=[f"{i}\n({int(v):,})" for i, v in valid.items()],
                autopct="%1.1f%%", colors=colors5[: len(valid)], startangle=90,
                textprops={"fontsize": 9},
            )
            ax.set_title("Distribusi Segmen Pelanggan (RFM)", fontsize=12, fontweight="bold")
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)

        with colR:
            st.markdown("##### Ringkasan Rata-rata per Segmen")
            summary_tbl = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
            summary_tbl = summary_tbl.reindex([s for s in order_seg if s in summary_tbl.index])
            st.dataframe(summary_tbl, use_container_width=True)
            st.caption(
                "**Champions/Loyal**: baru bertransaksi, sering, & nilai tinggi — pertahankan dengan loyalty program.  \n"
                "**Potential Loyalist**: berpotensi naik kelas dengan sedikit dorongan promosi.  \n"
                "**At Risk**: sudah lama tidak bertransaksi — perlu kampanye reaktivasi.  \n"
                "**Hibernating**: risiko churn tertinggi — pertimbangkan diskon khusus win-back."
            )

        with st.expander("Lihat data RFM per pelanggan"):
            st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True)

# ======================================================================
# TAB 4: Geospatial
# ======================================================================
if active_tab == TAB_LABELS[3]:
    st.subheader("Analisis Lanjutan: Geospatial Analysis")
    st.markdown("Distribusi revenue & rata-rata review score pelanggan berdasarkan **state (provinsi)** di Brasil.")

    state_agg = (
        df.groupby("customer_state")
        .agg(
            revenue=("order_value", "sum"),
            n_customers=("customer_unique_id", "nunique"),
            n_orders=("order_id", "nunique"),
            avg_review=("review_score", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    state_agg["lat"] = state_agg["customer_state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[0])
    state_agg["lng"] = state_agg["customer_state"].map(lambda s: STATE_COORDS.get(s, (np.nan, np.nan))[1])

    total_revenue_geo = state_agg["revenue"].sum()
    top5_share = state_agg.head(5)["revenue"].sum() / total_revenue_geo * 100 if total_revenue_geo > 0 else 0

    if FOLIUM_AVAILABLE:
        st.markdown("##### 🗺️ Peta Interaktif: Revenue & Review Score per State")

        use_choropleth = st.checkbox(
            "Tampilkan sebagai peta Choropleth (garis batas wilayah) + bubble",
            value=False,
            help=(
                "Mode Choropleth mewarnai tiap wilayah state berdasarkan revenue, "
                "dan tetap menampilkan bubble (ukuran = jumlah pelanggan, warna = review score) "
                "di atasnya — jadi 3 variabel tetap terlihat sekaligus. Membutuhkan data batas "
                "wilayah (GeoJSON) yang diambil dari internet; jika gagal, otomatis kembali ke mode bubble biasa."
            ),
        )

        map_df = state_agg.dropna(subset=["lat", "lng"]).copy()
        if map_df.empty:
            st.warning("Tidak ada koordinat state yang cocok untuk filter saat ini.")
        else:
            valid_review = map_df["avg_review"].dropna()
            vmin = float(valid_review.min()) if not valid_review.empty else 3.5
            vmax = float(valid_review.max()) if not valid_review.empty else 5.0
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
            review_colormap = bcm.LinearColormap(
                colors=["#C1272D", "#FDD835", "#2E7D32"], vmin=vmin, vmax=vmax,
                caption="Rata-rata Review Score (bubble)",
            )

            geojson = load_brazil_geojson() if use_choropleth else None
            if use_choropleth and geojson is None:
                st.warning(
                    "Data batas wilayah (GeoJSON) tidak berhasil diambil/dicocokkan (butuh koneksi "
                    "internet ke GitHub). Menampilkan mode bubble map biasa sebagai gantinya."
                )

            m = folium.Map(location=[-14.2, -51.9], zoom_start=4, tiles=ESRI_TILES, attr=ESRI_ATTR)

            if use_choropleth and geojson is not None:
                st.caption(
                    "**Warna wilayah (choropleth)** = total revenue • **ukuran bubble** = jumlah pelanggan unik • "
                    "**warna bubble** = rata-rata review score. Klik/hover untuk detail. "
                    "*Membutuhkan koneksi internet untuk memuat tile peta & data batas wilayah.*"
                )
                folium.Choropleth(
                    geo_data=geojson,
                    data=map_df,
                    columns=["customer_state", "revenue"],
                    key_on="feature.properties._state_code",
                    fill_color="YlOrRd",
                    fill_opacity=0.7,
                    line_opacity=0.4,
                    line_color="white",
                    legend_name="Total Revenue (R$) per State",
                    name="Choropleth Revenue",
                    nan_fill_color="#DDDDDD",
                ).add_to(m)

                bubble_layer = folium.FeatureGroup(name="Bubble: Pelanggan & Review Score")
                max_cust = map_df["n_customers"].max()
                for _, row in map_df.iterrows():
                    radius = 4 + (row["n_customers"] / max_cust) * 22 if max_cust > 0 else 6
                    color = review_colormap(row["avg_review"]) if pd.notna(row["avg_review"]) else "#9E9E9E"
                    review_txt = f"{row['avg_review']:.2f}" if pd.notna(row["avg_review"]) else "N/A"
                    popup_html = (
                        f"<b>{row['customer_state']}</b><br>"
                        f"Revenue: R$ {row['revenue']:,.0f}<br>"
                        f"Pelanggan unik: {row['n_customers']:,}<br>"
                        f"Jumlah order: {row['n_orders']:,}<br>"
                        f"Avg Review: {review_txt}"
                    )
                    folium.CircleMarker(
                        location=[row["lat"], row["lng"]],
                        radius=radius,
                        color="#222222",
                        weight=1,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        tooltip=f"{row['customer_state']}: {row['n_customers']:,} pelanggan",
                        popup=folium.Popup(popup_html, max_width=250),
                    ).add_to(bubble_layer)
                bubble_layer.add_to(m)
                review_colormap.add_to(m)
                folium.LayerControl(collapsed=True).add_to(m)
            else:
                st.caption(
                    "Ukuran bubble = revenue, warna bubble = rata-rata review score (merah = rendah, hijau = tinggi). "
                    "Klik/hover tiap bubble untuk detail. *Membutuhkan koneksi internet untuk memuat tile peta.*"
                )
                max_rev = map_df["revenue"].max()
                for _, row in map_df.iterrows():
                    radius = 6 + (row["revenue"] / max_rev) * 34 if max_rev > 0 else 8
                    color = review_colormap(row["avg_review"]) if pd.notna(row["avg_review"]) else "#9E9E9E"
                    review_txt = f"{row['avg_review']:.2f}" if pd.notna(row["avg_review"]) else "N/A"
                    popup_html = (
                        f"<b>{row['customer_state']}</b><br>"
                        f"Revenue: R$ {row['revenue']:,.0f}<br>"
                        f"Pelanggan unik: {row['n_customers']:,}<br>"
                        f"Jumlah order: {row['n_orders']:,}<br>"
                        f"Avg Review: {review_txt}"
                    )
                    folium.CircleMarker(
                        location=[row["lat"], row["lng"]],
                        radius=radius,
                        color=color,
                        weight=1,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.75,
                        tooltip=f"{row['customer_state']}: R$ {row['revenue']:,.0f}",
                        popup=folium.Popup(popup_html, max_width=250),
                    ).add_to(m)
                review_colormap.add_to(m)

            st_folium(m, use_container_width=True, height=480, returned_objects=[])
    else:
        st.warning(
            "Paket `folium` / `streamlit-folium` belum terpasang, dashboard menampilkan versi "
            "fallback (scatter plot) sebagai gantinya. Jalankan `pip install -r requirements.txt` "
            "lalu restart dashboard untuk mendapatkan peta interaktif."
        )
        fig_geo, ax = plt.subplots(figsize=(8, 8))
        map_df = state_agg.dropna(subset=["lat", "lng"])
        sizes = (map_df["revenue"] / map_df["revenue"].max()) * 3000 + 50 if map_df["revenue"].max() > 0 else 100
        sc = ax.scatter(
            map_df["lng"], map_df["lat"], s=sizes, c=map_df["avg_review"],
            cmap="RdYlGn", alpha=0.75, edgecolors="black", linewidths=0.5,
        )
        for _, row in map_df.sort_values("revenue", ascending=False).head(10).iterrows():
            ax.annotate(row["customer_state"], (row["lng"], row["lat"]), fontsize=9, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points")
        plt.colorbar(sc, ax=ax, label="Rata-rata Review Score")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Distribusi Geografis Revenue per State (fallback, tanpa basemap)", fontsize=12, fontweight="bold")
        fig_geo.tight_layout()
        st.pyplot(fig_geo)
        plt.close(fig_geo)

    st.markdown("---")
    top_n = st.slider("Jumlah state ditampilkan pada grafik batang", 5, 27, 10, key="n_state")

    colL, colR = st.columns([1.1, 1])
    with colL:
        fig6, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
        top_states = state_agg.head(top_n)
        ax.barh(top_states["customer_state"][::-1], (top_states["revenue"] / 1e6)[::-1], color=sns.color_palette("crest", top_n))
        ax.set_xlabel("Revenue (Juta R$)")
        ax.set_title(f"Top {top_n} State Berdasarkan Revenue", fontsize=12, fontweight="bold")
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    with colR:
        st.markdown("##### Tabel Ringkasan per State")
        st.dataframe(
            state_agg[["customer_state", "revenue", "n_customers", "n_orders", "avg_review"]].round(2),
            use_container_width=True,
            height=420,
        )

    st.info(f"📌 **5 state teratas** menyumbang **{top5_share:.1f}%** dari total revenue pada filter saat ini.")

st.markdown("---")
st.caption(
    "Sumber data: E-Commerce Public Dataset (Olist Brazilian E-Commerce) • "
)