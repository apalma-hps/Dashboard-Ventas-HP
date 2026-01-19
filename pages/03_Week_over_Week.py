# pages/03_Week_over_Week.py
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
import re
import unicodedata
import json
import urllib.request
import matplotlib  # (lo dejas si lo usas en otra parte)


# =========================================================
# CONFIG BÁSICA (DEBE SER LO PRIMERO EN STREAMLIT)
# =========================================================
st.set_page_config(
    page_title="Week over Week – Marcas HP",
    page_icon="📈",
    layout="wide",
)

st.sidebar.markdown("### Actualización")
if st.sidebar.button("🔄 Actualizar data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Última vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


# =========================================================
# TEMA ALTAIR
# =========================================================
def byf_altair_theme():
    return {
        "config": {
            "background": "rgba(0,0,0,0)",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": "#1B1D22", "titleColor": "#1B1D22"},
            "legend": {"labelColor": "#1B1D22", "titleColor": "#1B1D22"},
            "range": {
                "category": [
                    "#1B1D22",
                    "#7AD9CF",
                    "#A7F0E3",
                    "#B8EDEA",
                    "#6F7277",
                    "#37D2A3",
                ],
            },
        }
    }


alt.themes.register("byf_theme", byf_altair_theme)
alt.themes.enable("byf_theme")


# =========================================================
# ESTILOS (NO DEPENDE DE DATA)
# =========================================================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 80% 10%, #A7F0E3 0, #F5F8F9 40%),
                    radial-gradient(circle at 0% 80%, #B8EDEA 0, #F5F8F9 45%),
                    #F5F8F9;
    }
    [data-testid="stHeader"] { background: transparent; }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* KPIs */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.90);
        border-radius: 18px;
        padding: 1rem 1.3rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
        text-align: center;
    }

    .stDataFrame {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 0.3rem 0.3rem 0.8rem 0.3rem;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    .comparison-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        box-shadow:0 12px 30px rgba(15, 23, 42, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin-bottom: 1rem;
    }

    .period-label {
        font-size: 0.85rem;
        color: #6F7277;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .story-section {
        background: rgba(255, 255, 255, 0.95);
        border-left: 4px solid #7AD9CF;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }

    .story-section h4 {
        color: #1B1D22;
        margin-top: 0;
    }

    /* =====================================================
       🆕 Resumen Ejecutivo Profesional (nuevo)
       ===================================================== */
    .exec-summary-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.30);
        border-left: 6px solid #7AD9CF;
        margin: 1rem 0;
    }

    .exec-title {
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6F7277;
    }

    .exec-main-metric {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin: 0.8rem 0 1.2rem 0;
        flex-wrap: wrap;
    }

    .exec-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B1D22;
    }

    .exec-change {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        background: rgba(245, 248, 249, 0.90);
    }

    .exec-change.positive { color: #37D2A3; }
    .exec-change.negative { color: #f5576c; }

    .exec-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }

    @media (max-width: 1100px) {
        .exec-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 700px) {
        .exec-grid { grid-template-columns: 1fr; }
    }

    .exec-metric-box {
        background: rgba(245, 248, 249, 0.50);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .exec-metric-label {
        font-size: 0.82rem;
        color: #6F7277;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }

    .exec-metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B1D22;
        margin-bottom: 0.3rem;
    }

    .exec-metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }

    .exec-metric-delta.pos { color: #37D2A3; }
    .exec-metric-delta.neg { color: #f5576c; }
    .exec-metric-delta.neutral { color: #6F7277; }

    /* =====================================================
       🆕 Insights Discretos (nuevo)
       ===================================================== */
    .insight-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin: 0.9rem 0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-left: 4px solid #7AD9CF;
    }

    .insight-card.alert { border-left-color: #f5576c; }
    .insight-card.success { border-left-color: #37D2A3; }

    .insight-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.7rem;
    }

    .insight-icon { font-size: 1.2rem; }

    .insight-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1B1D22;
        margin: 0;
    }

    .insight-body {
        font-size: 0.92rem;
        color: #6F7277;
        line-height: 1.6;
        margin: 0 0 0.8rem 0;
    }

    .insight-action {
        background: rgba(122, 217, 207, 0.08);
        border-left: 3px solid #7AD9CF;
        padding: 0.7rem 0.9rem;
        border-radius: 8px;
        font-size: 0.88rem;
    }

    .insight-action strong {
        color: #1B1D22;
        display: block;
        margin-bottom: 0.3rem;
    }

    .insight-action p {
        margin: 0;
        color: #6F7277;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER (LOGO + TÍTULO)
# =========================================================
LOGO_URL = "https://raw.githubusercontent.com/apalma-hps/Dashboard-Ventas-HP/main/logo_hp.png"

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown(
        f"""
        <div style="
            width:120px;height:120px;
            border-radius:60px;
            border:4px solid #7AD9CF;
            display:flex;align-items:center;justify-content:center;
            background-color:#F5F8F9;
            box-shadow:0 18px 45px rgba(15,23,42,0.15);">
            <img src="{LOGO_URL}" style="width:70%;height:70%;border-radius:50%;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_title:
    st.markdown(
        """
        <h1 style="margin-bottom:0;">Análisis Week over Week – Marcas HP</h1>
        <p style="color:#6F7277;font-size:0.95rem;margin-top:0.25rem;">
        Comparativas semanales y 4 semanas vs 4 semanas · KPIs de performance · Insights accionables
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# =========================================================
# CONFIG / CONSTANTES
# =========================================================
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"
APPSCRIPT_URL = st.secrets.get("APPSCRIPT_URL", "").strip()

COL_CC = "Restaurante"
COL_ESTADO = "Estado"
COL_FECHA = "Fecha"
COL_SUBTOT = "Subtotal"
COL_TOTAL = "Total"
COL_DESCUENTOS = "Descuentos"
COL_TIPO = "Tipo"
COL_FOLIO = "Folio"
COL_VENTAS = "ventas_efectivas"
COL_DETALLE = "Detalle Items"


# =========================================================
# HELPERS GENERALES
# =========================================================
def fmt_money(x):
    return "—" if (x is None or pd.isna(x)) else f"${x:,.0f}"


def fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{x * 100:,.1f}%"


def fmt_pp(x):
    return "—" if (x is None or pd.isna(x)) else f"{x * 100:+.1f}pp"


def fmt_change_ratio(x):
    if x is None or pd.isna(x):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x * 100:,.1f}%"


def html_no_md_codeblock(s: str) -> str:
    """
    FIX: Streamlit usa Markdown para render; si una línea inicia con >=4 espacios,
    se interpreta como code-block y el HTML se imprime.
    Este helper elimina indentación al inicio de cada línea.
    """
    if s is None:
        return ""
    s = s.replace("\u00A0", " ").replace("\t", " ")
    return "\n".join(line.lstrip(" ") for line in s.splitlines()).strip()


def clean_money_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def is_delivery(val):
    try:
        return "delivery" in str(val).lower()
    except Exception:
        return False


def to_monday(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.to_datetime(d)
    return d - timedelta(days=d.weekday())


def safe_pct_change(current, previous):
    if isinstance(previous, (pd.Series, np.ndarray)) or isinstance(current, (pd.Series, np.ndarray)):
        cur = pd.to_numeric(current, errors="coerce")
        prev = pd.to_numeric(previous, errors="coerce")
        out = (cur - prev) / prev
        out = out.where((prev != 0) & (~prev.isna()) & (~cur.isna()), np.nan)
        return out

    if previous is None or pd.isna(previous) or previous == 0:
        return None
    if current is None or pd.isna(current):
        return None
    return (current - previous) / previous


def filtrar_periodo(df, inicio, fin, restaurante=None):
    mask = (df[COL_FECHA].dt.date >= inicio.date()) & (df[COL_FECHA].dt.date <= fin.date())
    out = df[mask].copy()
    if restaurante and restaurante != "Todos los restaurantes":
        out = out[out[COL_CC] == restaurante]
    return out


def detect_tax_column(df_: pd.DataFrame):
    candidates = ["Impuestos", "IVA", "Tax", "Taxes", "Impuesto", "VAT"]
    for c in candidates:
        if c in df_.columns:
            return c
    return None


def get_void_mask(df_: pd.DataFrame, col_estado: str) -> pd.Series:
    if col_estado in df_.columns:
        return df_[col_estado].astype(str).str.strip().str.lower().eq("void")
    return pd.Series(False, index=df_.index)


def post_json(url: str, payload: dict, timeout=20):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# =========================================================
# MIX HELPERS
# =========================================================
def norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    s = re.sub(r"[^\w\s\/\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\-+", "-", s).strip()
    return s


def _parse_base_item_with_price(raw: str):
    txt = str(raw or "").strip()
    if not txt:
        return "", 0, None

    precio_linea = None
    m_precio = re.search(r"\(\s*\$?\s*([\d\.,]+)\s*\)", txt)
    if m_precio:
        num = m_precio.group(1).replace(",", "")
        try:
            precio_linea = float(num)
        except ValueError:
            precio_linea = None
        txt = txt[:m_precio.start()].strip()

    qty = 1
    m_qty = re.search(r"\s+[xX]\s*(\d+)\s*$", txt)
    if m_qty:
        qty = int(m_qty.group(1))
        txt = txt[:m_qty.start()].strip()

    nombre = txt.strip()
    if not nombre or qty <= 0:
        return "", 0, None

    precio_unit = None
    if precio_linea is not None:
        precio_unit = precio_linea / qty if qty else None

    return nombre, qty, precio_unit


def parse_detalle_items_base_y_complementos_precio(texto: str):
    registros = []
    if not isinstance(texto, str) or not texto.strip():
        return registros

    productos_principales = [p.strip() for p in texto.split("|") if p.strip()]

    for producto in productos_principales:
        if "[" in producto and "]" in producto:
            partes = producto.split("[", 1)
            base_texto = partes[0].strip()
            complementos_texto = partes[1].split("]")[0].strip()
        else:
            base_texto = producto.strip()
            complementos_texto = ""

        nombre_base, qty_base, precio_unit = _parse_base_item_with_price(base_texto)
        if nombre_base and qty_base > 0:
            registros.append({
                "item": nombre_base,
                "qty": qty_base,
                "tipo_concepto": "base",
                "precio_unitario": precio_unit,
            })

        if complementos_texto:
            complementos = [c.strip() for c in complementos_texto.split(",") if c.strip()]
            for comp in complementos:
                comp_limpio = comp.lstrip("+").strip()
                if comp_limpio:
                    registros.append({
                        "item": comp_limpio,
                        "qty": 1,
                        "tipo_concepto": "complemento",
                        "precio_unitario": None,
                    })

    return registros


@st.cache_data(ttl=600)
def load_catalogo_conceptos():
    try:
        cat = pd.read_csv(CATALOGO_URL)
    except Exception as e:
        st.warning(f"No se pudo cargar el catálogo: {e}")
        return None

    cat.columns = [c.strip() for c in cat.columns]
    required = {"concepto", "tipo_concepto", "Categoria"}
    if not required.issubset(set(cat.columns)):
        st.warning("El catálogo debe tener columnas: concepto, tipo_concepto, Categoria.")
        return None

    cat["concepto"] = cat["concepto"].astype(str).str.strip()
    cat["tipo_concepto"] = cat["tipo_concepto"].astype(str).str.strip().str.lower()
    cat["Categoria_raw"] = cat["Categoria"].astype(str).str.strip()

    is_instr = cat["Categoria_raw"].str.match(r"(?i)^\s*contar\s+")
    m = cat["Categoria_raw"].str.extract(r"(?i)^\s*contar\s*(?:como\s+)?(.+?)\s*$")

    cat["concepto_canonico"] = np.where(is_instr, m[0].str.strip(), cat["concepto"])
    cat["Clasificación"] = np.where(is_instr, "REMAP", cat["Categoria_raw"])

    cat["concepto_key"] = cat["concepto_canonico"].map(norm_key)
    cat["categoria_mix"] = cat["Categoria_raw"].astype(str).str.strip()

    return cat


def calcular_mix_ventas_dinero(df_periodo: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    if df_periodo is None or df_periodo.empty:
        return pd.DataFrame(columns=["categoria_mix", "ventas_estimadas", "mix_pct", "tickets_con_categoria"])

    if COL_DETALLE not in df_periodo.columns:
        return pd.DataFrame(columns=["categoria_mix", "ventas_estimadas", "mix_pct", "tickets_con_categoria"])

    regs = []
    for _, row in df_periodo.iterrows():
        items = parse_detalle_items_base_y_complementos_precio(row.get(COL_DETALLE, ""))
        for item in items:
            item["folio"] = row.get(COL_FOLIO, "")
        regs.extend(items)

    if not regs:
        return pd.DataFrame(columns=["categoria_mix", "ventas_estimadas", "mix_pct", "tickets_con_categoria"])

    flat = pd.DataFrame(regs)
    flat["item_key"] = flat["item"].map(norm_key)

    j = flat.merge(
        catalogo,
        left_on="item_key",
        right_on="concepto_key",
        how="left",
        suffixes=("", "_cat"),
    )

    j = j[j["concepto"].notna()].copy()

    clas = j["Clasificación"].fillna("").astype(str).str.strip().str.lower()
    j = j[(clas != "no contar") & (clas != "remap")].copy()

    j["precio_unitario"] = pd.to_numeric(j["precio_unitario"], errors="coerce")
    j = j[j["tipo_concepto"].astype(str).str.strip().str.lower().eq("base")].copy()

    j["ventas_estimadas"] = (j["qty"].fillna(0).astype(float) * j["precio_unitario"]).fillna(0.0)

    if float(j["ventas_estimadas"].sum()) <= 0:
        return pd.DataFrame(columns=["categoria_mix", "ventas_estimadas", "mix_pct", "tickets_con_categoria"])

    j["categoria_mix"] = j["categoria_mix"].fillna("Sin categoría").astype(str).str.strip()

    out = (
        j.groupby("categoria_mix", as_index=False)
        .agg(
            ventas_estimadas=("ventas_estimadas", "sum"),
            tickets_con_categoria=("folio", "nunique"),
        )
        .sort_values("ventas_estimadas", ascending=False)
    )

    total = float(out["ventas_estimadas"].sum())
    out["mix_pct"] = out["ventas_estimadas"] / total if total > 0 else 0.0
    return out


# =========================================================
# ✅ (FIX) COMPOSICIÓN PROMEDIO DE ORDEN: DEFINIDA ANTES DE USARSE
# =========================================================
def calcular_composicion_promedio_orden(df_periodo, catalogo):
    """
    Composición promedio de orden:
    - items_por_orden: cuántos items (base) de la categoría aparecen por orden en promedio
    - penetracion: % de órdenes donde aparece la categoría
    """
    if df_periodo is None or df_periodo.empty or catalogo is None or COL_DETALLE not in df_periodo.columns:
        return pd.DataFrame()

    is_void = get_void_mask(df_periodo, COL_ESTADO)
    df_valido = df_periodo[~is_void].copy()
    if df_valido.empty:
        return pd.DataFrame()

    all_items = []
    for _, row in df_valido.iterrows():
        items = parse_detalle_items_base_y_complementos_precio(row.get(COL_DETALLE, ""))
        for item in items:
            item["folio"] = row.get(COL_FOLIO, "")
        all_items.extend(items)

    if not all_items:
        return pd.DataFrame()

    flat = pd.DataFrame(all_items)
    flat["item_key"] = flat["item"].map(norm_key)

    j = flat.merge(
        catalogo,
        left_on="item_key",
        right_on="concepto_key",
        how="left",
        suffixes=("", "_cat"),
    )
    j = j[j["concepto"].notna()].copy()

    clas = j["Clasificación"].fillna("").astype(str).str.strip().str.lower()
    j = j[(clas != "no contar") & (clas != "remap")].copy()
    j["categoria_mix"] = j["categoria_mix"].fillna("Sin categoría").astype(str).str.strip()

    j_base = j[j["tipo_concepto"].astype(str).str.strip().str.lower().eq("base")].copy()

    total_tickets = df_valido[COL_FOLIO].nunique() if COL_FOLIO in df_valido.columns else 0
    if total_tickets <= 0 or j_base.empty:
        return pd.DataFrame()

    composicion = (
        j_base.groupby("categoria_mix", as_index=False)
        .agg(
            total_items=("qty", "sum"),
            tickets_con_categoria=("folio", "nunique"),
        )
    )

    composicion["items_por_orden"] = composicion["total_items"] / total_tickets
    composicion["penetracion"] = composicion["tickets_con_categoria"] / total_tickets
    composicion = composicion.sort_values("items_por_orden", ascending=False)
    return composicion


# =========================================================
# MÉTRICAS
# =========================================================
def calcular_metricas(df_periodo: pd.DataFrame):
    if df_periodo is None or df_periodo.empty:
        return {
            "ventas": 0.0,
            "tickets": 0,
            "cancelados": 0,
            "ticket_promedio": 0.0,
            "ventas_delivery": 0.0,
            "pct_delivery": 0.0,
            "orders_per_day": 0,
            "items_por_ticket": 0.0,
            "dias_operados": 0,
        }

    is_void = get_void_mask(df_periodo, COL_ESTADO)

    ventas = float(df_periodo[COL_VENTAS].sum()) if COL_VENTAS in df_periodo.columns else 0.0

    tickets = int(df_periodo.loc[~is_void, COL_FOLIO].nunique()) if COL_FOLIO in df_periodo.columns else 0
    cancelados = int(df_periodo.loc[is_void, COL_FOLIO].nunique()) if COL_FOLIO in df_periodo.columns else int(is_void.sum())

    ticket_promedio = float(ventas / tickets) if tickets > 0 else 0.0

    if COL_TIPO in df_periodo.columns and COL_VENTAS in df_periodo.columns:
        ventas_delivery = float(df_periodo.loc[df_periodo[COL_TIPO].map(is_delivery), COL_VENTAS].sum())
    else:
        ventas_delivery = 0.0

    pct_delivery = float(ventas_delivery / ventas) if ventas > 0 else 0.0

    dias_unicos = int(df_periodo[COL_FECHA].dt.date.nunique()) if COL_FECHA in df_periodo.columns else 0
    orders_per_day = int(round(tickets / dias_unicos)) if dias_unicos > 0 else 0

    items_por_ticket = 0.0
    if COL_DETALLE in df_periodo.columns and tickets > 0:
        total_items = 0
        for _, row in df_periodo[~is_void].iterrows():
            items = parse_detalle_items_base_y_complementos_precio(row.get(COL_DETALLE, ""))
            total_items += sum(item["qty"] for item in items if item["tipo_concepto"] == "base")
        items_por_ticket = total_items / tickets if tickets > 0 else 0.0

    return {
        "ventas": ventas,
        "tickets": tickets,
        "cancelados": cancelados,
        "ticket_promedio": ticket_promedio,
        "ventas_delivery": ventas_delivery,
        "pct_delivery": pct_delivery,
        "orders_per_day": orders_per_day,
        "items_por_ticket": items_por_ticket,
        "dias_operados": dias_unicos,
    }


def delta_color(change, invert: bool = False):
    if change is None or pd.isna(change):
        return "off"
    return "inverse" if invert else "normal"


# =========================================================
# RESUMEN EJECUTIVO + INSIGHTS
# =========================================================
def build_resumen_ejecutivo_profesional(metricas_act, metricas_ant, mix_act, mix_ant):
    cambio_ventas = safe_pct_change(metricas_act["ventas"], metricas_ant["ventas"])
    cambio_tickets = safe_pct_change(metricas_act["tickets"], metricas_ant["tickets"])
    cambio_ticket_prom = safe_pct_change(metricas_act["ticket_promedio"], metricas_ant["ticket_promedio"])
    cambio_items = metricas_act["items_por_ticket"] - metricas_ant["items_por_ticket"]
    cambio_delivery_pp = (metricas_act["pct_delivery"] - metricas_ant["pct_delivery"])

    clase_cambio = "positive" if (cambio_ventas or 0) >= 0 else "negative"

    total_orders = (metricas_act["tickets"] + metricas_act["cancelados"])
    tasa_cancel = (metricas_act["cancelados"] / total_orders) if total_orders > 0 else 0.0

    top_gan_txt = ""
    top_per_txt = ""
    if mix_act is not None and mix_ant is not None and not mix_act.empty and not mix_ant.empty:
        comp = mix_act.merge(mix_ant, on="categoria_mix", how="outer", suffixes=("_act", "_ant")).fillna(0)
        comp["delta_mix"] = comp["mix_pct_act"] - comp["mix_pct_ant"]
        comp = comp.sort_values("delta_mix", ascending=False)
        if len(comp) > 0:
            gan = comp.iloc[0]
            if float(gan["delta_mix"]) > 0.02:
                top_gan_txt = f"{gan['categoria_mix']} {fmt_pp(gan['delta_mix'])}"
            per = comp.iloc[-1]
            if float(per["delta_mix"]) < -0.02:
                top_per_txt = f"{per['categoria_mix']} {fmt_pp(per['delta_mix'])}"

    return {
        "ventas": metricas_act["ventas"],
        "cambio_ventas": cambio_ventas if cambio_ventas is not None else 0,
        "clase_cambio": clase_cambio,
        "tickets": metricas_act["tickets"],
        "cambio_tickets": cambio_tickets if cambio_tickets is not None else 0,
        "ticket_prom": metricas_act["ticket_promedio"],
        "cambio_ticket_prom": cambio_ticket_prom if cambio_ticket_prom is not None else 0,
        "items_orden": metricas_act["items_por_ticket"],
        "cambio_items": cambio_items,
        "pct_delivery": metricas_act["pct_delivery"],
        "cambio_delivery_pp": cambio_delivery_pp,
        "tasa_cancel": tasa_cancel,
        "top_ganador": top_gan_txt,
        "top_perdedor": top_per_txt,
    }


def render_resumen_ejecutivo(data):
    ventas_fmt = fmt_money(data["ventas"])
    cambio_ventas_fmt = fmt_change_ratio(data["cambio_ventas"])
    tickets_fmt = f"{data['tickets']:,}"
    cambio_tickets_fmt = fmt_change_ratio(data["cambio_tickets"])
    ticket_prom_fmt = fmt_money(data["ticket_prom"])
    cambio_ticket_prom_fmt = fmt_change_ratio(data["cambio_ticket_prom"])
    items_orden_fmt = f"{data['items_orden']:.1f}"
    cambio_items_fmt = f"{data['cambio_items']:+.1f}"
    pct_delivery_fmt = fmt_pct(data["pct_delivery"])
    cambio_delivery_fmt = fmt_pp(data["cambio_delivery_pp"])
    tasa_cancel_fmt = fmt_pct(data["tasa_cancel"])
    tasa_cancel_label = "Alta" if data["tasa_cancel"] > 0.05 else "Normal"

    tickets_class = "pos" if data["cambio_tickets"] >= 0 else "neg"
    ticket_prom_class = "pos" if data["cambio_ticket_prom"] >= 0 else "neg"
    items_class = "pos" if data["cambio_items"] >= 0 else "neg"
    delivery_class = "pos" if data["cambio_delivery_pp"] >= 0 else "neg"

    top_ganador_display = f"↑ {data['top_ganador']}" if data["top_ganador"] else "—"
    top_perdedor_display = f"↓ {data['top_perdedor']}" if data["top_perdedor"] else "—"

    html = f"""
<div class="exec-summary-card">
    <div class="exec-title">Resumen Ejecutivo Semanal</div>

    <div class="exec-main-metric">
        <div class="exec-value">{ventas_fmt}</div>
        <div class="exec-change {data['clase_cambio']}">{cambio_ventas_fmt}</div>
    </div>

    <div class="exec-grid">
        <div class="exec-metric-box">
            <div class="exec-metric-label">Tickets</div>
            <div class="exec-metric-value">{tickets_fmt}</div>
            <div class="exec-metric-delta {tickets_class}">{cambio_tickets_fmt}</div>
        </div>

        <div class="exec-metric-box">
            <div class="exec-metric-label">Ticket Promedio</div>
            <div class="exec-metric-value">{ticket_prom_fmt}</div>
            <div class="exec-metric-delta {ticket_prom_class}">{cambio_ticket_prom_fmt}</div>
        </div>

        <div class="exec-metric-box">
            <div class="exec-metric-label">Items por Orden</div>
            <div class="exec-metric-value">{items_orden_fmt}</div>
            <div class="exec-metric-delta {items_class}">{cambio_items_fmt}</div>
        </div>

        <div class="exec-metric-box">
            <div class="exec-metric-label">% Delivery</div>
            <div class="exec-metric-value">{pct_delivery_fmt}</div>
            <div class="exec-metric-delta {delivery_class}">{cambio_delivery_fmt}</div>
        </div>

        <div class="exec-metric-box">
            <div class="exec-metric-label">Tasa Cancelación</div>
            <div class="exec-metric-value">{tasa_cancel_fmt}</div>
            <div class="exec-metric-delta neutral">{tasa_cancel_label}</div>
        </div>

        <div class="exec-metric-box">
            <div class="exec-metric-label">Mix Destacado</div>
            <div class="exec-metric-value" style="font-size:0.95rem;">{top_ganador_display}</div>
            <div class="exec-metric-delta neutral" style="font-size:0.80rem;">{top_perdedor_display}</div>
        </div>
    </div>
</div>
"""
    st.markdown(html_no_md_codeblock(html), unsafe_allow_html=True)


def generar_insights_accionables(metricas_act, metricas_ant, mix_act, mix_ant):
    insights = []

    cambio_ventas = safe_pct_change(metricas_act["ventas"], metricas_ant["ventas"])
    if cambio_ventas is not None:
        if cambio_ventas > 0.05:
            insights.append({
                "tipo": "success",
                "icon": "📈",
                "titulo": "Crecimiento Semanal Positivo",
                "mensaje": (
                    f"Las ventas crecieron {fmt_change_ratio(cambio_ventas)} "
                    f"({fmt_money(metricas_ant['ventas'])} → {fmt_money(metricas_act['ventas'])}), "
                    f"con {metricas_act['tickets'] - metricas_ant['tickets']:+,} tickets adicionales."
                ),
                "accion": (
                    "Identificar el driver principal (tráfico vs ticket). "
                    "Si fue tráfico, reforzar adquisición. Si fue ticket, estandarizar script de upselling y combos."
                )
            })
        elif cambio_ventas < -0.03:
            insights.append({
                "tipo": "alert",
                "icon": "⚠️",
                "titulo": "Contracción Semanal",
                "mensaje": f"Las ventas cayeron {fmt_change_ratio(cambio_ventas)} ({fmt_money(metricas_ant['ventas'])} → {fmt_money(metricas_act['ventas'])}).",
                "accion": (
                    "Revisar: (1) operación (faltantes/cierres), (2) competencia/promos, (3) calendario. "
                    "Definir 1 palanca prioritaria para recuperar en 7 días."
                )
            })

    cambio_ticket = safe_pct_change(metricas_act["ticket_promedio"], metricas_ant["ticket_promedio"])
    if cambio_ticket is not None and abs(cambio_ticket) > 0.04:
        if cambio_ticket > 0:
            insights.append({
                "tipo": "success",
                "icon": "💰",
                "titulo": "Mejora en Ticket Promedio",
                "mensaje": f"El ticket promedio aumentó {fmt_change_ratio(cambio_ticket)} (de {fmt_money(metricas_ant['ticket_promedio'])} a {fmt_money(metricas_act['ticket_promedio'])}).",
                "accion": "Reforzar combos y visibilidad de premium. Incentivos simples por upsell + monitoreo diario de ticket promedio."
            })
        else:
            insights.append({
                "tipo": "alert",
                "icon": "📉",
                "titulo": "Ticket Promedio en Descenso",
                "mensaje": f"El ticket promedio bajó {fmt_change_ratio(cambio_ticket)} (de {fmt_money(metricas_ant['ticket_promedio'])} a {fmt_money(metricas_act['ticket_promedio'])}).",
                "accion": "Auditar disponibilidad de combos, ejecución en caja y productos ancla. Si hay faltantes, ajustar par e inventario."
            })

    if mix_act is not None and mix_ant is not None and not mix_act.empty and not mix_ant.empty:
        comp = mix_act.merge(mix_ant, on="categoria_mix", how="outer", suffixes=("_act", "_ant")).fillna(0)
        comp["cambio_mix"] = comp["mix_pct_act"] - comp["mix_pct_ant"]
        comp = comp.sort_values("cambio_mix", ascending=False)

        if len(comp) > 0:
            gan = comp.iloc[0]
            if float(gan["cambio_mix"]) > 0.03:
                insights.append({
                    "tipo": "success",
                    "icon": "🌟",
                    "titulo": f"'{gan['categoria_mix']}' en Auge",
                    "mensaje": f"Ganó {fmt_pp(gan['cambio_mix'])} de participación ({fmt_pct(gan['mix_pct_ant'])} → {fmt_pct(gan['mix_pct_act'])}).",
                    "accion": "Asegurar inventario y crear 1 combo protagonista con esta categoría. Darle visibilidad en menú y caja."
                })

            per = comp.iloc[-1]
            if float(per["cambio_mix"]) < -0.03:
                insights.append({
                    "tipo": "alert",
                    "icon": "📊",
                    "titulo": f"'{per['categoria_mix']}' Perdiendo Terreno",
                    "mensaje": f"Perdió {fmt_pp(abs(per['cambio_mix']))} ({fmt_pct(per['mix_pct_ant'])} → {fmt_pct(per['mix_pct_act'])}).",
                    "accion": "Diagnóstico rápido: calidad/consistencia, precio vs competencia, visibilidad. Si no es estratégica, replantear propuesta."
                })

    if metricas_act["cancelados"] > 0:
        total_orders = (metricas_act["tickets"] + metricas_act["cancelados"])
        tasa = (metricas_act["cancelados"] / total_orders) if total_orders > 0 else 0.0
        if tasa > 0.05:
            insights.append({
                "tipo": "alert",
                "icon": "🚫",
                "titulo": "Tasa de Cancelación Elevada",
                "mensaje": f"{metricas_act['cancelados']} órdenes canceladas ({fmt_pct(tasa)} del total).",
                "accion": "Reducir a <3%: checklist de caja, control de agotados, y límites de tiempo de producción en picos."
            })

    cambio_delivery = safe_pct_change(metricas_act["ventas_delivery"], metricas_ant["ventas_delivery"])
    if cambio_delivery is not None and abs(cambio_delivery) > 0.15:
        if cambio_delivery > 0:
            insights.append({
                "tipo": "neutral",
                "icon": "🚗",
                "titulo": "Canal Delivery en Crecimiento",
                "mensaje": f"Ventas por delivery {fmt_change_ratio(cambio_delivery)} ({fmt_pct(metricas_act['pct_delivery'])} del total).",
                "accion": "Optimizar tiempos, empaque y fotos/descripciones. Subir rating y reducir reclamos."
            })
        else:
            insights.append({
                "tipo": "alert",
                "icon": "🚗",
                "titulo": "Canal Delivery en Contracción",
                "mensaje": f"Ventas por delivery {fmt_change_ratio(cambio_delivery)} ({fmt_pct(metricas_act['pct_delivery'])} del total).",
                "accion": "Revisar: visibilidad en apps, pricing, promos, tiempos, y disponibilidad de productos top."
            })

    if metricas_act["items_por_ticket"] > 0:
        valor_item = (metricas_act["ticket_promedio"] / metricas_act["items_por_ticket"]) if metricas_act["items_por_ticket"] > 0 else 0
        insights.append({
            "tipo": "neutral",
            "icon": "🛒",
            "titulo": "Anatomía de Orden Típica",
            "mensaje": f"Cada orden: {metricas_act['items_por_ticket']:.1f} items × {fmt_money(valor_item)}/item = {fmt_money(metricas_act['ticket_promedio'])}.",
            "accion": f"Meta operativa: llevar items/orden a {metricas_act['items_por_ticket'] * 1.15:.1f} con sugerencia activa + combos prearmados."
        })

    return insights


def render_insights(insights):
    if not insights:
        st.info("No hay insights destacados para este periodo. La operación se mantiene estable.")
        return

    for ins in insights:
        tipo_class = "success" if ins["tipo"] == "success" else ("alert" if ins["tipo"] == "alert" else "")

        accion_html = ""
        if ins.get("accion"):
            accion_html = f"""
            <div class="insight-action">
                <strong>💡 Acción Recomendada</strong>
                <p>{ins['accion']}</p>
            </div>
            """

        insight_html = f"""
<div class="insight-card {tipo_class}">
    <div class="insight-header">
        <div class="insight-icon">{ins['icon']}</div>
        <div class="insight-title">{ins['titulo']}</div>
    </div>
    <div class="insight-body">{ins['mensaje']}</div>
    {accion_html}
</div>
"""
        st.markdown(html_no_md_codeblock(insight_html), unsafe_allow_html=True)


# =========================================================
# CARGA Y LIMPIEZA DE DATOS
# =========================================================
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_URL)
    df_.columns = [c.strip() for c in df_.columns]
    return df_


df = load_data()

required_cols = {COL_CC, COL_FECHA, COL_TOTAL}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en la base: {missing}")
    st.stop()

df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
df[COL_TOTAL] = clean_money_series(df[COL_TOTAL]).fillna(0.0)

if COL_SUBTOT not in df.columns:
    st.warning("No existe 'Subtotal'. Se usará Subtotal = Total como fallback.")
    df[COL_SUBTOT] = df[COL_TOTAL]
else:
    df[COL_SUBTOT] = clean_money_series(df[COL_SUBTOT]).fillna(0.0)

if COL_DESCUENTOS in df.columns:
    df[COL_DESCUENTOS] = clean_money_series(df[COL_DESCUENTOS]).fillna(0.0)
else:
    df[COL_DESCUENTOS] = 0.0

tax_col = detect_tax_column(df)
if tax_col is not None:
    df["_impuestos"] = clean_money_series(df[tax_col]).fillna(0.0)
else:
    df["_impuestos"] = (df[COL_TOTAL] - df[COL_SUBTOT]).clip(lower=0.0)

is_void_all = get_void_mask(df, COL_ESTADO)

df["_calc_sti_d"] = (df[COL_SUBTOT] + df["_impuestos"] - df[COL_DESCUENTOS]).fillna(0.0)
df["_ventas_brutas_regla"] = np.where(df["_calc_sti_d"] > df[COL_TOTAL], df[COL_TOTAL], df["_calc_sti_d"])

df[COL_VENTAS] = np.where(
    is_void_all,
    0.0,
    pd.Series(df["_ventas_brutas_regla"], index=df.index).clip(lower=0.0),
)

df = df[df[COL_FECHA].notna()].copy()
if df.empty:
    st.info("No hay datos con fecha válida.")
    st.stop()


# =========================================================
# FILTROS
# =========================================================
st.sidebar.markdown("### Selección de Periodo")

fecha_max = df[COL_FECHA].max()
fecha_default = to_monday(fecha_max)

fecha_seleccionada = st.sidebar.date_input(
    "Semana de referencia (se toma el lunes de esa semana)",
    value=fecha_default.date(),
    help="Selecciona cualquier día. Se ajusta automáticamente al lunes.",
)

inicio_semana_sel = to_monday(pd.to_datetime(fecha_seleccionada))

st.sidebar.markdown("---")

restaurantes = ["Todos los restaurantes"] + sorted(df[COL_CC].dropna().unique().tolist())
rest_seleccionado = st.sidebar.selectbox("Restaurante", restaurantes, index=0)


# =========================================================
# PERIODOS
# =========================================================
semana_actual_inicio = inicio_semana_sel
semana_actual_fin = semana_actual_inicio + timedelta(days=6)

semana_anterior_inicio = semana_actual_inicio - timedelta(days=7)
semana_anterior_fin = semana_anterior_inicio + timedelta(days=6)

cuatro_sem_actual_inicio = semana_actual_inicio - timedelta(days=21)
cuatro_sem_actual_fin = semana_actual_fin

cuatro_sem_anterior_inicio = cuatro_sem_actual_inicio - timedelta(days=28)
cuatro_sem_anterior_fin = cuatro_sem_actual_inicio - timedelta(days=1)


# =========================================================
# DATA POR PERIODO
# =========================================================
df_sem_actual = filtrar_periodo(df, semana_actual_inicio, semana_actual_fin, rest_seleccionado)
df_sem_anterior = filtrar_periodo(df, semana_anterior_inicio, semana_anterior_fin, rest_seleccionado)
df_4sem_actual = filtrar_periodo(df, cuatro_sem_actual_inicio, cuatro_sem_actual_fin, rest_seleccionado)
df_4sem_anterior = filtrar_periodo(df, cuatro_sem_anterior_inicio, cuatro_sem_anterior_fin, rest_seleccionado)

metricas_sem_actual = calcular_metricas(df_sem_actual)
metricas_sem_anterior = calcular_metricas(df_sem_anterior)
metricas_4sem_actual = calcular_metricas(df_4sem_actual)
metricas_4sem_anterior = calcular_metricas(df_4sem_anterior)


# =========================================================
# CATÁLOGO + MIX + COMPOSICIÓN
# =========================================================
catalogo = load_catalogo_conceptos()

mix_sem = pd.DataFrame()
mix_sem_ant = pd.DataFrame()
mix_4 = pd.DataFrame()
composicion_sem = pd.DataFrame()

if catalogo is not None and COL_DETALLE in df.columns:
    mix_sem = calcular_mix_ventas_dinero(df_sem_actual, catalogo)
    mix_sem_ant = calcular_mix_ventas_dinero(df_sem_anterior, catalogo)
    mix_4 = calcular_mix_ventas_dinero(df_4sem_actual, catalogo)
    composicion_sem = calcular_composicion_promedio_orden(df_sem_actual, catalogo)


# =========================================================
# HEADER: INFO DE PERIODOS
# =========================================================
st.markdown(f"### Análisis : **{rest_seleccionado}**")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""
        <div class="comparison-card">
            <div class="period-label">Week over Week (WoW)</div>
            <div style="font-size:0.9rem;color:#6F7277;margin-bottom:0.5rem;">
                <strong>Semana Actual:</strong> {semana_actual_inicio.strftime('%d/%m/%Y')} - {semana_actual_fin.strftime('%d/%m/%Y')}
            </div>
            <div style="font-size:0.9rem;color:#6F7277;">
                <strong>Semana Anterior:</strong> {semana_anterior_inicio.strftime('%d/%m/%Y')} - {semana_anterior_fin.strftime('%d/%m/%Y')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="comparison-card">
            <div class="period-label">4 Weeks vs 4 Weeks (4WoW)</div>
            <div style="font-size:0.9rem;color:#6F7277;margin-bottom:0.5rem;">
                <strong>4 Semanas Actuales:</strong> {cuatro_sem_actual_inicio.strftime('%d/%m/%Y')} - {cuatro_sem_actual_fin.strftime('%d/%m/%Y')}
            </div>
            <div style="font-size:0.9rem;color:#6F7277;">
                <strong>4 Semanas Anteriores:</strong> {cuatro_sem_anterior_inicio.strftime('%d/%m/%Y')} - {cuatro_sem_anterior_fin.strftime('%d/%m/%Y')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# =========================================================
# ✅ RESUMEN EJECUTIVO (NUEVO)
# =========================================================
st.markdown("## 📊 Resumen Ejecutivo")

data_exec = build_resumen_ejecutivo_profesional(
    metricas_sem_actual, metricas_sem_anterior,
    mix_sem, mix_sem_ant
)
render_resumen_ejecutivo(data_exec)

st.markdown("---")

st.markdown("## 💡 Insights Accionables")
insights = generar_insights_accionables(
    metricas_sem_actual, metricas_sem_anterior,
    mix_sem, mix_sem_ant
)
render_insights(insights)

st.markdown("---")


# =========================================================
# KPIs WoW
# =========================================================
st.markdown("### Week over Week (WoW)")

cambio_ventas_wow = safe_pct_change(metricas_sem_actual["ventas"], metricas_sem_anterior["ventas"])
cambio_tickets_wow = safe_pct_change(metricas_sem_actual["tickets"], metricas_sem_anterior["tickets"])
cambio_ticket_prom_wow = safe_pct_change(metricas_sem_actual["ticket_promedio"], metricas_sem_anterior["ticket_promedio"])
cambio_orders_day_wow = safe_pct_change(metricas_sem_actual["orders_per_day"], metricas_sem_anterior["orders_per_day"])
cambio_cancelados_wow = safe_pct_change(metricas_sem_actual["cancelados"], metricas_sem_anterior["cancelados"])
cambio_items_wow = metricas_sem_actual["items_por_ticket"] - metricas_sem_anterior["items_por_ticket"]
cambio_pct_delivery_wow_pp = (metricas_sem_actual["pct_delivery"] - metricas_sem_anterior["pct_delivery"])

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Ventas",
        fmt_money(metricas_sem_actual["ventas"]),
        delta=f"{fmt_change_ratio(cambio_ventas_wow)} vs sem. anterior",
        delta_color=delta_color(cambio_ventas_wow, invert=False),
    )

with col2:
    st.metric(
        "Tickets",
        f"{metricas_sem_actual['tickets']:,.0f}",
        delta=f"{fmt_change_ratio(cambio_tickets_wow)} vs sem. anterior",
        delta_color=delta_color(cambio_tickets_wow, invert=False),
    )

with col3:
    st.metric(
        "Ticket Prom.",
        fmt_money(metricas_sem_actual["ticket_promedio"]),
        delta=f"{fmt_change_ratio(cambio_ticket_prom_wow)} vs sem. anterior",
        delta_color=delta_color(cambio_ticket_prom_wow, invert=False),
    )

with col4:
    st.metric(
        "Items/Orden",
        f"{metricas_sem_actual['items_por_ticket']:.1f}",
        delta=f"{cambio_items_wow:+.1f} items",
        delta_color="normal" if cambio_items_wow >= 0 else "inverse",
    )

with col5:
    st.metric(
        "Órdenes/día",
        f"{metricas_sem_actual['orders_per_day']:,.0f}",
        delta=f"{fmt_change_ratio(cambio_orders_day_wow)}",
        delta_color=delta_color(cambio_orders_day_wow, invert=False),
    )

with col6:
    st.metric(
        "% Delivery",
        fmt_pct(metricas_sem_actual["pct_delivery"]),
        delta=f"{fmt_pp(cambio_pct_delivery_wow_pp)}",
        delta_color="normal" if cambio_pct_delivery_wow_pp >= 0 else "inverse",
    )

st.markdown("#### 📋 Detalle Comparativo WoW")
wow_data = pd.DataFrame({
    "Métrica": ["Ventas", "Tickets", "Cancelados", "Ticket Promedio", "Items/Orden", "Órdenes/día", "Ventas Delivery", "% Delivery"],
    "Semana Anterior": [
        fmt_money(metricas_sem_anterior["ventas"]),
        f"{metricas_sem_anterior['tickets']:,.0f}",
        f"{metricas_sem_anterior['cancelados']:,.0f}",
        fmt_money(metricas_sem_anterior["ticket_promedio"]),
        f"{metricas_sem_anterior['items_por_ticket']:.1f}",
        f"{metricas_sem_anterior['orders_per_day']:,.0f}",
        fmt_money(metricas_sem_anterior["ventas_delivery"]),
        fmt_pct(metricas_sem_anterior["pct_delivery"]),
    ],
    "Semana Actual": [
        fmt_money(metricas_sem_actual["ventas"]),
        f"{metricas_sem_actual['tickets']:,.0f}",
        f"{metricas_sem_actual['cancelados']:,.0f}",
        fmt_money(metricas_sem_actual["ticket_promedio"]),
        f"{metricas_sem_actual['items_por_ticket']:.1f}",
        f"{metricas_sem_actual['orders_per_day']:,.0f}",
        fmt_money(metricas_sem_actual["ventas_delivery"]),
        fmt_pct(metricas_sem_actual["pct_delivery"]),
    ],
    "Cambio": [
        fmt_change_ratio(cambio_ventas_wow),
        fmt_change_ratio(cambio_tickets_wow),
        fmt_change_ratio(cambio_cancelados_wow),
        fmt_change_ratio(cambio_ticket_prom_wow),
        f"{cambio_items_wow:+.1f}",
        fmt_change_ratio(cambio_orders_day_wow),
        fmt_change_ratio(safe_pct_change(metricas_sem_actual["ventas_delivery"], metricas_sem_anterior["ventas_delivery"])),
        fmt_pp(cambio_pct_delivery_wow_pp),
    ],
})
st.dataframe(wow_data.set_index("Métrica"), use_container_width=True)

st.markdown("---")


# =========================================================
# COMPOSICIÓN PROMEDIO DE ORDEN (VISUAL)
# =========================================================
if composicion_sem is not None and (not composicion_sem.empty):
    st.markdown("## 🛒 Composición Promedio de una Orden")

    st.markdown(
        """
        <div class="story-section">
            <h4>¿Qué hay en una orden típica?</h4>
            <p>Este análisis muestra cuántos items de cada categoría se incluyen en promedio en cada orden,
            y en qué porcentaje de órdenes aparece cada categoría. Úsalo para:</p>
            <ul>
                <li><strong>Diseñar combos</strong> que reflejen el comportamiento real del cliente</li>
                <li><strong>Identificar oportunidades</strong> de cross-selling entre categorías complementarias</li>
                <li><strong>Fijar metas</strong> realistas de items por orden</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_display = composicion_sem.copy().rename(columns={
        "categoria_mix": "Categoría",
        "items_por_orden": "Items/Orden Promedio",
        "penetracion": "% Órdenes con Categoría",
        "total_items": "Total Items Vendidos",
    })

    st.dataframe(
        comp_display.style.format({
            "Items/Orden Promedio": "{:.2f}",
            "% Órdenes con Categoría": lambda x: fmt_pct(x),
            "Total Items Vendidos": "{:,.0f}",
        }),
        use_container_width=True,
    )

    if len(comp_display) > 0:
        chart_comp = alt.Chart(composicion_sem).mark_bar().encode(
            x=alt.X("items_por_orden:Q", title="Items por Orden Promedio"),
            y=alt.Y("categoria_mix:N", sort="-x", title="Categoría"),
            color=alt.Color("items_por_orden:Q", scale=alt.Scale(scheme="teals"), legend=None),
            tooltip=[
                alt.Tooltip("categoria_mix:N", title="Categoría"),
                alt.Tooltip("items_por_orden:Q", title="Items/Orden", format=".2f"),
                alt.Tooltip("penetracion:Q", title="Penetración", format=".1%"),
            ],
        ).properties(
            title="Items por Orden Promedio por Categoría",
            height=400,
        )
        st.altair_chart(chart_comp, use_container_width=True)

st.markdown("---")


# =========================================================
# MIX DE VENTAS
# =========================================================
st.markdown("## 🥘 Mix de Ventas por Categoría")

if catalogo is None:
    st.warning("No se pudo cargar el catálogo. No es posible calcular el mix.")
elif COL_DETALLE not in df.columns:
    st.warning(f"No existe la columna '{COL_DETALLE}'. Sin esa columna no podemos mapear conceptos a categorías.")
else:
    if (mix_sem is not None and mix_sem_ant is not None) and (not mix_sem.empty) and (not mix_sem_ant.empty):
        st.markdown("### 📊 Evolución del Mix (WoW)")

        mix_comp = mix_sem.merge(
            mix_sem_ant,
            on="categoria_mix",
            how="outer",
            suffixes=("_actual", "_ant"),
        ).fillna(0)

        mix_comp["cambio_pct"] = (mix_comp["mix_pct_actual"] - mix_comp["mix_pct_ant"])
        mix_comp["cambio_ventas"] = safe_pct_change(mix_comp["ventas_estimadas_actual"], mix_comp["ventas_estimadas_ant"])
        mix_comp = mix_comp.sort_values("ventas_estimadas_actual", ascending=False)

        mix_display = mix_comp[[
            "categoria_mix",
            "ventas_estimadas_ant",
            "mix_pct_ant",
            "ventas_estimadas_actual",
            "mix_pct_actual",
            "cambio_pct",
            "cambio_ventas",
        ]].copy()

        mix_display.columns = [
            "Categoría",
            "Ventas (Sem Ant)",
            "Mix % (Sem Ant)",
            "Ventas (Sem Act)",
            "Mix % (Sem Act)",
            "Cambio Mix (pp)",
            "Cambio Ventas %",
        ]

        st.dataframe(
            mix_display.style.format({
                "Ventas (Sem Ant)": lambda x: fmt_money(x),
                "Mix % (Sem Ant)": lambda x: fmt_pct(x),
                "Ventas (Sem Act)": lambda x: fmt_money(x),
                "Mix % (Sem Act)": lambda x: fmt_pct(x),
                "Cambio Mix (pp)": lambda x: fmt_pp(x),
                "Cambio Ventas %": lambda x: fmt_change_ratio(x),
            }).background_gradient(
                subset=["Cambio Mix (pp)"],
                cmap="RdYlGn",
                vmin=-0.1,
                vmax=0.1,
            ),
            use_container_width=True,
        )

        chart_data = mix_comp.head(10).copy()

        chart_mix = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("cambio_pct:Q", title="Cambio en Participación (pp)", axis=alt.Axis(format="%")),
            y=alt.Y("categoria_mix:N", sort="-x", title="Categoría"),
            color=alt.condition(
                alt.datum.cambio_pct > 0,
                alt.value("#37D2A3"),
                alt.value("#f5576c"),
            ),
            tooltip=[
                alt.Tooltip("categoria_mix:N", title="Categoría"),
                alt.Tooltip("cambio_pct:Q", title="Cambio Mix", format="+.1%"),
                alt.Tooltip("cambio_ventas:Q", title="Cambio Ventas", format="+.1%"),
            ],
        ).properties(
            title="Cambio en Participación de Mix (Top 10 Categorías)",
            height=400,
        )
        st.altair_chart(chart_mix, use_container_width=True)

    if mix_4 is not None and (not mix_4.empty):
        st.markdown("### 📈 Mix 4 Semanas Actuales")

        mix_4_display = mix_4.copy()
        mix_4_display["valor_promedio_categoria"] = (
            mix_4_display["ventas_estimadas"] / mix_4_display["tickets_con_categoria"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        mix_4_display = mix_4_display.rename(columns={
            "categoria_mix": "Categoría",
            "ventas_estimadas": "Ventas Estimadas",
            "mix_pct": "Participación %",
            "tickets_con_categoria": "Tickets con Categoría",
            "valor_promedio_categoria": "Valor Prom. por Ticket",
        })

        st.dataframe(
            mix_4_display.style.format({
                "Ventas Estimadas": lambda x: fmt_money(x),
                "Participación %": lambda x: fmt_pct(x),
                "Tickets con Categoría": "{:,.0f}",
                "Valor Prom. por Ticket": lambda x: fmt_money(x),
            }),
            use_container_width=True,
        )

st.markdown("---")


# =========================================================
# METAS SUGERIDAS
# =========================================================
st.markdown("## 🎯 Metas Sugeridas")

def generar_metas_sugeridas(metricas_actual, mix_actual, composicion_actual):
    metas = []

    if metricas_actual["ventas"] > 0:
        incremento_sugerido = 0.10
        meta_ventas = metricas_actual["ventas"] * (1 + incremento_sugerido)
        metas.append({
            "categoria": "Ventas Totales",
            "actual": fmt_money(metricas_actual["ventas"]),
            "meta_sugerida": fmt_money(meta_ventas),
            "incremento": fmt_pct(incremento_sugerido),
            "palanca": "Aumentar tráfico y ticket promedio",
        })

    if metricas_actual["ticket_promedio"] > 0:
        incremento_ticket = 0.05
        meta_ticket = metricas_actual["ticket_promedio"] * (1 + incremento_ticket)
        metas.append({
            "categoria": "Ticket Promedio",
            "actual": fmt_money(metricas_actual["ticket_promedio"]),
            "meta_sugerida": fmt_money(meta_ticket),
            "incremento": fmt_pct(incremento_ticket),
            "palanca": "Upselling y combos estratégicos",
        })

    if mix_actual is not None and (not mix_actual.empty):
        top_cat = mix_actual.iloc[0]
        incremento_mix = 0.03
        meta_mix = float(top_cat["mix_pct"]) + incremento_mix
        metas.append({
            "categoria": f"Mix de '{top_cat['categoria_mix']}'",
            "actual": fmt_pct(float(top_cat["mix_pct"])),
            "meta_sugerida": fmt_pct(meta_mix),
            "incremento": fmt_pp(incremento_mix),
            "palanca": "Promoción activa y visibilidad en menú",
        })

    if metricas_actual["items_por_ticket"] > 0:
        incremento_items = 0.15
        meta_items = metricas_actual["items_por_ticket"] * (1 + incremento_items)
        metas.append({
            "categoria": "Items por Orden",
            "actual": f"{metricas_actual['items_por_ticket']:.1f}",
            "meta_sugerida": f"{meta_items:.1f}",
            "incremento": fmt_pct(incremento_items),
            "palanca": "Sugerencias del personal y combos",
        })

    return pd.DataFrame(metas)


metas_df = generar_metas_sugeridas(metricas_sem_actual, mix_sem, composicion_sem)

mensaje_meta = ""
if metas_df is not None and (not metas_df.empty):
    st.markdown(
        """
        <div class="story-section">
            <h4>Metas basadas en datos</h4>
            <p>Las siguientes metas están calibradas con base en el desempeño actual y las mejores prácticas del sector restaurantero.
            Son <strong>ambiciosas pero alcanzables</strong> con la ejecución correcta.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(metas_df, use_container_width=True)

    st.markdown("### 📝 Mensaje para el equipo (incluir en correo)")
    mensaje_meta = st.text_area(
        "Mensaje adicional",
        value=f"""Meta para la próxima semana:

1. Incrementar ventas en 10% mediante mayor tráfico y mejora en ticket promedio
2. Aumentar items por orden de {metricas_sem_actual['items_por_ticket']:.1f} a {metricas_sem_actual['items_por_ticket'] * 1.15:.1f} mediante sugerencias activas
3. Reforzar las categorías top en el mix y recuperar las que han perdido participación
4. Reducir cancelaciones mediante mejor capacitación y control de calidad

Acciones clave:
- Capacitar al personal en técnicas de upselling
- Crear combos basados en la composición real de órdenes
- Promocionar categorías de alto margen que han mostrado crecimiento
- Monitorear diariamente el ticket promedio y ajustar estrategias""",
        height=250,
    )

st.markdown("---")


# =========================================================
# ENVÍO DE EMAIL
# =========================================================
st.markdown("## 📧 Enviar Reporte por Correo")

if not APPSCRIPT_URL:
    st.info("Configura `APPSCRIPT_URL` en `.streamlit/secrets.toml` para habilitar el envío.")
else:
    payload = {
        "restaurante": rest_seleccionado,
        "periodos": {
            "semana_actual": f"{semana_actual_inicio.strftime('%d/%m/%Y')} - {semana_actual_fin.strftime('%d/%m/%Y')}",
            "semana_anterior": f"{semana_anterior_inicio.strftime('%d/%m/%Y')} - {semana_anterior_fin.strftime('%d/%m/%Y')}",
            "cuatro_sem_actual": f"{cuatro_sem_actual_inicio.strftime('%d/%m/%Y')} - {cuatro_sem_actual_fin.strftime('%d/%m/%Y')}",
            "cuatro_sem_anterior": f"{cuatro_sem_anterior_inicio.strftime('%d/%m/%Y')} - {cuatro_sem_anterior_fin.strftime('%d/%m/%Y')}",
        },
        "resumen_ejecutivo": {
            "ventas_actuales": metricas_sem_actual["ventas"],
            "cambio_wow": safe_pct_change(metricas_sem_actual["ventas"], metricas_sem_anterior["ventas"]),
            "ticket_promedio": metricas_sem_actual["ticket_promedio"],
            "items_por_orden": metricas_sem_actual["items_por_ticket"],
            "insights": [{"titulo": i["titulo"], "mensaje": i["mensaje"]} for i in insights[:5]] if insights else [],
        },
        "meta": {
            "mensaje": mensaje_meta or "",
        },
        "mix_semana": mix_sem.head(15).to_dict(orient="records") if mix_sem is not None and not mix_sem.empty else [],
        "mix_4sem": mix_4.head(15).to_dict(orient="records") if mix_4 is not None and not mix_4.empty else [],
        "composicion_orden": composicion_sem.head(10).to_dict(orient="records") if composicion_sem is not None and not composicion_sem.empty else [],
        "metas": metas_df.to_dict(orient="records") if metas_df is not None and not metas_df.empty else [],
    }

    if st.button("📩 Enviar reporte ejecutivo"):
        try:
            with st.spinner("Enviando reporte..."):
                resp = post_json(APPSCRIPT_URL, payload)
                if resp.get("ok"):
                    st.success(f"✅ Correo enviado exitosamente a: {resp.get('to', '—')}")
                    st.balloons()
                else:
                    st.error(f"❌ Error en Apps Script: {resp.get('error')}")
        except Exception as e:
            st.error(f"❌ No se pudo contactar Apps Script: {e}")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#6F7277;font-size:0.85rem;padding:2rem 0;">
        Dashboard Week over Week · Marcas HP · Powered by Streamlit & Altair
    </div>
    """,
    unsafe_allow_html=True,
)
