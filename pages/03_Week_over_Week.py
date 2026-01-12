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


# ============= CONFIG BÁSICA =============
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


# ===== Tema de Altair =====
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


# ===== Estilos =====
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

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.90);
        border-radius: 18px;
        padding: 1rem 1.3rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== Logo + título =====
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
        Comparativas semanales y 4 semanas vs 4 semanas · KPIs de performance
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# =========================================================
# URLs Y HELPERS
# =========================================================
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"

# Catálogo de conceptos (mismo que usas en Health Rate)
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"

# Apps Script (solo envío de correo). Mejor ponerlo en secrets aunque no sea secreto.
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

# 👇 Esta columna es clave para el conteo (como en Health Rate)
COL_DETALLE = "Detalle Items"


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


def detect_tax_column(df_: pd.DataFrame) -> str | None:
    candidates = ["Impuestos", "IVA", "Tax", "Taxes", "Impuesto", "VAT"]
    for c in candidates:
        if c in df_.columns:
            return c
    return None


def get_void_mask(df_: pd.DataFrame, col_estado: str) -> pd.Series:
    if col_estado in df_.columns:
        return df_[col_estado].astype(str).str.strip().str.lower().eq("void")
    return pd.Series(False, index=df_.index)


# =========================================================
# UPSSELL HELPERS (misma lógica del Health Rate)
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


def _parse_base_item(raw: str):
    txt = str(raw or "").strip()
    if not txt:
        return "", 0

    # quitar "( $123 )" si existe
    txt = re.sub(r"\(\s*\$?\s*[\d\.,]+\s*\)", "", txt).strip()

    qty = 1
    m_qty = re.search(r"\s+[xX]\s*(\d+)\s*$", txt)
    if m_qty:
        qty = int(m_qty.group(1))
        txt = txt[:m_qty.start()].strip()

    nombre = txt.strip()
    if not nombre:
        return "", 0
    return nombre, qty


def parse_detalle_items_base_y_complementos(texto: str):
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

        nombre_base, qty_base = _parse_base_item(base_texto)
        if nombre_base and qty_base > 0:
            registros.append({
                "item": nombre_base,
                "qty": qty_base,
                "tipo_concepto": "base",
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
                    })

    return registros


@st.cache_data(ttl=600)
def load_catalogo_conceptos() -> pd.DataFrame | None:
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

    cat["concepto_key"] = cat["concepto"].map(norm_key)

    return cat


def conteo_upsell(df_periodo: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    """
    Conteo real (unidades) de upsell en un periodo, basado en Detalle Items.
    Upsell = tipo_concepto del catálogo en {"bebidas","bebida","complementos","complemento"}.
    Excluye 'no contar' y 'REMAP'.
    """
    if df_periodo is None or df_periodo.empty:
        return pd.DataFrame(columns=["concepto_canonico", "tipo_concepto", "unidades"])

    if COL_DETALLE not in df_periodo.columns:
        return pd.DataFrame(columns=["concepto_canonico", "tipo_concepto", "unidades"])

    regs = []
    # iterrows está bien aquí; volumen por semana suele ser manejable.
    # Si luego quieres performance, lo vectorizamos.
    for _, row in df_periodo.iterrows():
        regs.extend(parse_detalle_items_base_y_complementos(row.get(COL_DETALLE, "")))

    if not regs:
        return pd.DataFrame(columns=["concepto_canonico", "tipo_concepto", "unidades"])

    flat = pd.DataFrame(regs)
    flat["item_key"] = flat["item"].map(norm_key)

    j = flat.merge(
        catalogo,
        left_on="item_key",
        right_on="concepto_key",
        how="left",
        suffixes=("", "_cat"),
    )

    # Solo mapeados
    j = j[j["concepto"].notna()].copy()

    j["Clasificación"] = j["Clasificación"].fillna("").astype(str).str.strip().str.lower()
    j = j[j["Clasificación"] != "no contar"]
    j = j[j["Clasificación"] != "remap"]
    j = j[j["Clasificación"] != "remap "]  # por si viene con espacio raro

    j["tipo_concepto"] = j["tipo_concepto"].fillna("").astype(str).str.strip().str.lower()
    j = j[j["tipo_concepto"].isin(["bebidas", "bebida", "complementos", "complemento"])].copy()

    j["concepto_canonico"] = j["concepto_canonico"].fillna(j["item"]).astype(str).str.strip()

    out = (
        j.groupby(["concepto_canonico", "tipo_concepto"], as_index=False)
        .agg(unidades=("qty", "sum"))
        .sort_values("unidades", ascending=False)
    )
    return out


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
        }

    is_void = get_void_mask(df_periodo, COL_ESTADO)

    ventas = float(df_periodo[COL_VENTAS].sum())

    # ✅ Tickets = folios NO VOID (no depende de ventas>0)
    tickets = int(df_periodo.loc[~is_void, COL_FOLIO].nunique()) if COL_FOLIO in df_periodo.columns else 0

    # ✅ Cancelados = folios VOID
    cancelados = int(df_periodo.loc[is_void, COL_FOLIO].nunique()) if COL_FOLIO in df_periodo.columns else int(is_void.sum())

    ticket_promedio = float(ventas / tickets) if tickets > 0 else 0.0

    if COL_TIPO in df_periodo.columns:
        ventas_delivery = float(df_periodo.loc[df_periodo[COL_TIPO].map(is_delivery), COL_VENTAS].sum())
    else:
        ventas_delivery = 0.0

    pct_delivery = float(ventas_delivery / ventas) if ventas > 0 else 0.0

    dias_unicos = int(df_periodo[COL_FECHA].dt.date.nunique())
    orders_per_day = int(round(tickets / dias_unicos)) if dias_unicos > 0 else 0

    return {
        "ventas": ventas,
        "tickets": tickets,
        "cancelados": cancelados,
        "ticket_promedio": ticket_promedio,
        "ventas_delivery": ventas_delivery,
        "pct_delivery": pct_delivery,
        "orders_per_day": orders_per_day,
    }


def delta_color(change, invert: bool = False):
    if change is None or pd.isna(change):
        return "off"
    return "inverse" if invert else "normal"


# =========================================================
# CARGA DE DATOS
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
    st.warning("No existe 'Subtotal'. Se usará Subtotal = Total como fallback (puede afectar precisión).")
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

is_void = get_void_mask(df, COL_ESTADO)

df["_calc_sti_d"] = (df[COL_SUBTOT] + df["_impuestos"] - df[COL_DESCUENTOS]).fillna(0.0)
df["_ventas_brutas_regla"] = np.where(df["_calc_sti_d"] > df[COL_TOTAL], df[COL_TOTAL], df["_calc_sti_d"])

df[COL_VENTAS] = np.where(is_void, 0.0, pd.Series(df["_ventas_brutas_regla"], index=df.index).clip(lower=0.0))

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
# KPIs WoW
# =========================================================
st.markdown("### Week over Week (WoW)")

cambio_ventas_wow = safe_pct_change(metricas_sem_actual["ventas"], metricas_sem_anterior["ventas"])
cambio_tickets_wow = safe_pct_change(metricas_sem_actual["tickets"], metricas_sem_anterior["tickets"])
cambio_ticket_prom_wow = safe_pct_change(metricas_sem_actual["ticket_promedio"], metricas_sem_anterior["ticket_promedio"])
cambio_orders_day_wow = safe_pct_change(metricas_sem_actual["orders_per_day"], metricas_sem_anterior["orders_per_day"])
cambio_cancelados_wow = safe_pct_change(metricas_sem_actual["cancelados"], metricas_sem_anterior["cancelados"])

cambio_ventas_delivery_wow = safe_pct_change(metricas_sem_actual["ventas_delivery"], metricas_sem_anterior["ventas_delivery"])
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
        "Cancelados",
        f"{metricas_sem_actual['cancelados']:,.0f}",
        delta=f"{fmt_change_ratio(cambio_cancelados_wow)} vs sem. anterior",
        delta_color=delta_color(cambio_cancelados_wow, invert=True),
    )

with col4:
    st.metric(
        "Ticket Prom.",
        fmt_money(metricas_sem_actual["ticket_promedio"]),
        delta=f"{fmt_change_ratio(cambio_ticket_prom_wow)} vs sem. anterior",
        delta_color=delta_color(cambio_ticket_prom_wow, invert=False),
    )

with col5:
    st.metric(
        "Órdenes / día",
        f"{metricas_sem_actual['orders_per_day']:,.0f}",
        delta=f"{fmt_change_ratio(cambio_orders_day_wow)} vs sem. anterior",
        delta_color=delta_color(cambio_orders_day_wow, invert=False),
    )

with col6:
    st.metric(
        "% Delivery",
        fmt_pct(metricas_sem_actual["pct_delivery"]),
        delta=f"{fmt_pp(cambio_pct_delivery_wow_pp)} vs sem. anterior",
        delta_color=delta_color(cambio_pct_delivery_wow_pp, invert=False),
    )

st.markdown("#### Comparativa Detallada WoW")
wow_data = pd.DataFrame({
    "Métrica": ["Ventas", "Tickets", "Cancelados", "Ticket Promedio", "Órdenes / día", "Ventas Delivery", "% Delivery"],
    "Semana Anterior": [
        fmt_money(metricas_sem_anterior["ventas"]),
        f"{metricas_sem_anterior['tickets']:,.0f}",
        f"{metricas_sem_anterior['cancelados']:,.0f}",
        fmt_money(metricas_sem_anterior["ticket_promedio"]),
        f"{metricas_sem_anterior['orders_per_day']:,.0f}",
        fmt_money(metricas_sem_anterior["ventas_delivery"]),
        fmt_pct(metricas_sem_anterior["pct_delivery"]),
    ],
    "Semana Actual": [
        fmt_money(metricas_sem_actual["ventas"]),
        f"{metricas_sem_actual['tickets']:,.0f}",
        f"{metricas_sem_actual['cancelados']:,.0f}",
        fmt_money(metricas_sem_actual["ticket_promedio"]),
        f"{metricas_sem_actual['orders_per_day']:,.0f}",
        fmt_money(metricas_sem_actual["ventas_delivery"]),
        fmt_pct(metricas_sem_actual["pct_delivery"]),
    ],
    "Cambio": [
        fmt_change_ratio(cambio_ventas_wow),
        fmt_change_ratio(cambio_tickets_wow),
        fmt_change_ratio(cambio_cancelados_wow),
        fmt_change_ratio(cambio_ticket_prom_wow),
        fmt_change_ratio(cambio_orders_day_wow),
        fmt_change_ratio(cambio_ventas_delivery_wow),
        fmt_pp(cambio_pct_delivery_wow_pp),
    ],
})
st.dataframe(wow_data.set_index("Métrica"), use_container_width=True)

st.markdown("---")


# =========================================================
# KPIs 4WoW
# =========================================================
st.markdown("### 4 Weeks vs 4 Weeks (4WoW)")

cambio_ventas_4wow = safe_pct_change(metricas_4sem_actual["ventas"], metricas_4sem_anterior["ventas"])
cambio_tickets_4wow = safe_pct_change(metricas_4sem_actual["tickets"], metricas_4sem_anterior["tickets"])
cambio_ticket_prom_4wow = safe_pct_change(metricas_4sem_actual["ticket_promedio"], metricas_4sem_anterior["ticket_promedio"])
cambio_orders_day_4wow = safe_pct_change(metricas_4sem_actual["orders_per_day"], metricas_4sem_anterior["orders_per_day"])

cambio_ventas_delivery_4wow = safe_pct_change(metricas_4sem_actual["ventas_delivery"], metricas_4sem_anterior["ventas_delivery"])
cambio_pct_delivery_4wow_pp = (metricas_4sem_actual["pct_delivery"] - metricas_4sem_anterior["pct_delivery"])

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Ventas",
        fmt_money(metricas_4sem_actual["ventas"]),
        delta=f"{fmt_change_ratio(cambio_ventas_4wow)} vs 4 sem. anteriores",
        delta_color=delta_color(cambio_ventas_4wow, invert=False),
    )

with col2:
    st.metric(
        "Tickets",
        f"{metricas_4sem_actual['tickets']:,.0f}",
        delta=f"{fmt_change_ratio(cambio_tickets_4wow)} vs 4 sem. anteriores",
        delta_color=delta_color(cambio_tickets_4wow, invert=False),
    )

with col3:
    st.metric(
        "Ticket Prom.",
        fmt_money(metricas_4sem_actual["ticket_promedio"]),
        delta=f"{fmt_change_ratio(cambio_ticket_prom_4wow)} vs 4 sem. anteriores",
        delta_color=delta_color(cambio_ticket_prom_4wow, invert=False),
    )

with col4:
    st.metric(
        "Órdenes / día",
        f"{metricas_4sem_actual['orders_per_day']:,.0f}",
        delta=f"{fmt_change_ratio(cambio_orders_day_4wow)} vs 4 sem. anteriores",
        delta_color=delta_color(cambio_orders_day_4wow, invert=False),
    )

with col5:
    st.metric(
        "% Delivery",
        fmt_pct(metricas_4sem_actual["pct_delivery"]),
        delta=f"{fmt_pp(cambio_pct_delivery_4wow_pp)} vs 4 sem. anteriores",
        delta_color=delta_color(cambio_pct_delivery_4wow_pp, invert=False),
    )

st.markdown("#### Comparativa Detallada 4WoW")
four_wow_data = pd.DataFrame({
    "Métrica": ["Ventas", "Tickets", "Ticket Promedio", "Órdenes / día", "Ventas Delivery", "% Delivery"],
    "4 Semanas Anteriores": [
        fmt_money(metricas_4sem_anterior["ventas"]),
        f"{metricas_4sem_anterior['tickets']:,.0f}",
        fmt_money(metricas_4sem_anterior["ticket_promedio"]),
        f"{metricas_4sem_anterior['orders_per_day']:,.0f}",
        fmt_money(metricas_4sem_anterior["ventas_delivery"]),
        fmt_pct(metricas_4sem_anterior["pct_delivery"]),
    ],
    "4 Semanas Actuales": [
        fmt_money(metricas_4sem_actual["ventas"]),
        f"{metricas_4sem_actual['tickets']:,.0f}",
        fmt_money(metricas_4sem_actual["ticket_promedio"]),
        f"{metricas_4sem_actual['orders_per_day']:,.0f}",
        fmt_money(metricas_4sem_actual["ventas_delivery"]),
        fmt_pct(metricas_4sem_actual["pct_delivery"]),
    ],
    "Cambio": [
        fmt_change_ratio(cambio_ventas_4wow),
        fmt_change_ratio(cambio_tickets_4wow),
        fmt_change_ratio(cambio_ticket_prom_4wow),
        fmt_change_ratio(cambio_orders_day_4wow),
        fmt_change_ratio(cambio_ventas_delivery_4wow),
        fmt_pp(cambio_pct_delivery_4wow_pp),
    ],
})
st.dataframe(four_wow_data.set_index("Métrica"), use_container_width=True)

st.markdown("---")


# =========================================================
# COMPARATIVAS POR RESTAURANTE (si "Todos")
# =========================================================
if rest_seleccionado == "Todos los restaurantes":
    st.markdown("---")
    st.markdown("### Comparativa por Restaurante (WoW y 4WoW)")

    restaurantes_lista = sorted(df[COL_CC].dropna().unique().tolist())

    comparativa_wow = []
    comparativa_4wow = []

    for rest in restaurantes_lista:
        df_rest_actual = filtrar_periodo(df, semana_actual_inicio, semana_actual_fin, rest)
        df_rest_anterior = filtrar_periodo(df, semana_anterior_inicio, semana_anterior_fin, rest)

        m_act = calcular_metricas(df_rest_actual)
        m_ant = calcular_metricas(df_rest_anterior)

        wow = {
            "Restaurante": rest,
            "Ventas (Actual)": m_act["ventas"],
            "Ventas (Anterior)": m_ant["ventas"],
            "Cambio Ventas %": safe_pct_change(m_act["ventas"], m_ant["ventas"]),
            "Tickets (Actual)": m_act["tickets"],
            "Tickets (Anterior)": m_ant["tickets"],
            "Cambio Tickets %": safe_pct_change(m_act["tickets"], m_ant["tickets"]),
            "Cancelados (Actual)": m_act["cancelados"],
            "Cancelados (Anterior)": m_ant["cancelados"],
            "Cambio Cancelados %": safe_pct_change(m_act["cancelados"], m_ant["cancelados"]),
            "Ticket Prom (Actual)": m_act["ticket_promedio"],
            "Ticket Prom (Anterior)": m_ant["ticket_promedio"],
            "Cambio Ticket Prom %": safe_pct_change(m_act["ticket_promedio"], m_ant["ticket_promedio"]),
            "% Delivery (Actual)": m_act["pct_delivery"],
            "% Delivery (Anterior)": m_ant["pct_delivery"],
            "Cambio % Delivery (pp)": (m_act["pct_delivery"] - m_ant["pct_delivery"]),
        }
        comparativa_wow.append(wow)

        df_rest_4act = filtrar_periodo(df, cuatro_sem_actual_inicio, cuatro_sem_actual_fin, rest)
        df_rest_4ant = filtrar_periodo(df, cuatro_sem_anterior_inicio, cuatro_sem_anterior_fin, rest)

        m4_act = calcular_metricas(df_rest_4act)
        m4_ant = calcular_metricas(df_rest_4ant)

        four = {
            "Restaurante": rest,
            "Ventas (4Act)": m4_act["ventas"],
            "Ventas (4Ant)": m4_ant["ventas"],
            "Cambio Ventas %": safe_pct_change(m4_act["ventas"], m4_ant["ventas"]),
            "Tickets (4Act)": m4_act["tickets"],
            "Tickets (4Ant)": m4_ant["tickets"],
            "Cambio Tickets %": safe_pct_change(m4_act["tickets"], m4_ant["tickets"]),
            "Cancelados (4Act)": m4_act["cancelados"],
            "Cancelados (4Ant)": m4_ant["cancelados"],
            "Cambio Cancelados %": safe_pct_change(m4_act["cancelados"], m4_ant["cancelados"]),
            "Ticket Prom (4Act)": m4_act["ticket_promedio"],
            "Ticket Prom (4Ant)": m4_ant["ticket_promedio"],
            "Cambio Ticket Prom %": safe_pct_change(m4_act["ticket_promedio"], m4_ant["ticket_promedio"]),
            "% Delivery (4Act)": m4_act["pct_delivery"],
            "% Delivery (4Ant)": m4_ant["pct_delivery"],
            "Cambio % Delivery (pp)": (m4_act["pct_delivery"] - m4_ant["pct_delivery"]),
        }
        comparativa_4wow.append(four)

    df_comp_wow = pd.DataFrame(comparativa_wow)
    df_comp_4wow = pd.DataFrame(comparativa_4wow)

    df_comp_wow = df_comp_wow.sort_values("Ventas (Actual)", ascending=False)

    st.markdown("#### WoW por Restaurante")
    view_wow = df_comp_wow[[
        "Restaurante",
        "Ventas (Anterior)", "Ventas (Actual)", "Cambio Ventas %",
        "Tickets (Anterior)", "Tickets (Actual)", "Cambio Tickets %",
        "Cancelados (Anterior)", "Cancelados (Actual)", "Cambio Cancelados %",
        "Ticket Prom (Anterior)", "Ticket Prom (Actual)", "Cambio Ticket Prom %",
        "% Delivery (Anterior)", "% Delivery (Actual)", "Cambio % Delivery (pp)",
    ]].copy()

    st.dataframe(
        view_wow.style.format({
            "Ventas (Anterior)": "${:,.0f}",
            "Ventas (Actual)": "${:,.0f}",
            "Cambio Ventas %": lambda x: fmt_change_ratio(x),

            "Tickets (Anterior)": "{:,.0f}",
            "Tickets (Actual)": "{:,.0f}",
            "Cambio Tickets %": lambda x: fmt_change_ratio(x),

            "Cancelados (Anterior)": "{:,.0f}",
            "Cancelados (Actual)": "{:,.0f}",
            "Cambio Cancelados %": lambda x: fmt_change_ratio(x),

            "Ticket Prom (Anterior)": "${:,.0f}",
            "Ticket Prom (Actual)": "${:,.0f}",
            "Cambio Ticket Prom %": lambda x: fmt_change_ratio(x),

            "% Delivery (Anterior)": lambda x: fmt_pct(x),
            "% Delivery (Actual)": lambda x: fmt_pct(x),
            "Cambio % Delivery (pp)": lambda x: fmt_pp(x),
        }),
        use_container_width=True,
    )

    st.markdown("---")
    df_comp_4wow = df_comp_4wow.sort_values("Ventas (4Act)", ascending=False)

    st.markdown("#### 4WoW por Restaurante")
    view_4wow = df_comp_4wow[[
        "Restaurante",
        "Ventas (4Ant)", "Ventas (4Act)", "Cambio Ventas %",
        "Tickets (4Ant)", "Tickets (4Act)", "Cambio Tickets %",
        "Cancelados (4Ant)", "Cancelados (4Act)", "Cambio Cancelados %",
        "Ticket Prom (4Ant)", "Ticket Prom (4Act)", "Cambio Ticket Prom %",
        "% Delivery (4Ant)", "% Delivery (4Act)", "Cambio % Delivery (pp)",
    ]].copy()

    st.dataframe(
        view_4wow.style.format({
            "Ventas (4Ant)": "${:,.0f}",
            "Ventas (4Act)": "${:,.0f}",
            "Cambio Ventas %": lambda x: fmt_change_ratio(x),

            "Tickets (4Ant)": "{:,.0f}",
            "Tickets (4Act)": "{:,.0f}",
            "Cambio Tickets %": lambda x: fmt_change_ratio(x),

            "Cancelados (4Ant)": "{:,.0f}",
            "Cancelados (4Act)": "{:,.0f}",
            "Cambio Cancelados %": lambda x: fmt_change_ratio(x),

            "Ticket Prom (4Ant)": "${:,.0f}",
            "Ticket Prom (4Act)": "${:,.0f}",
            "Cambio Ticket Prom %": lambda x: fmt_change_ratio(x),

            "% Delivery (4Ant)": lambda x: fmt_pct(x),
            "% Delivery (4Act)": lambda x: fmt_pct(x),
            "Cambio % Delivery (pp)": lambda x: fmt_pp(x),
        }),
        use_container_width=True,
    )


# =========================================================
# NUEVO: OPORTUNIDAD UPSELL + EMAIL (Apps Script solo envía)
# =========================================================
st.markdown("---")
st.markdown("## Oportunidad Upsell (bebidas + complementos)")

# Cargar catálogo
catalogo = load_catalogo_conceptos()
if catalogo is None:
    st.warning("No se pudo cargar el catálogo, por lo que no se puede calcular upsell.")
else:
    # Validar que exista Detalle Items
    if COL_DETALLE not in df.columns:
        st.warning(
            f"No existe la columna '{COL_DETALLE}' en tu CSV. "
            "Sin esa columna no se puede calcular upsell (no hay conceptos por pedido)."
        )
    else:
        # Conteos reales por periodo (SEM ACT y 4 SEM ACT)
        ups_sem = conteo_upsell(df_sem_actual, catalogo)
        ups_4 = conteo_upsell(df_4sem_actual, catalogo)

        unidades_sem = float(ups_sem["unidades"].sum()) if not ups_sem.empty else 0.0
        unidades_4 = float(ups_4["unidades"].sum()) if not ups_4.empty else 0.0

        tickets_sem = float(metricas_sem_actual["tickets"])
        tickets_4 = float(metricas_4sem_actual["tickets"])

        attach_real_sem = (unidades_sem / tickets_sem) if tickets_sem > 0 else 0.0
        attach_real_4 = (unidades_4 / tickets_4) if tickets_4 > 0 else 0.0

        cA, cB, cC = st.columns([1.2, 1.0, 2.8])
        with cA:
            attach_meta = st.number_input(
                "Meta attach rate upsell (%)",
                min_value=0.0, max_value=100.0,
                value=12.0, step=1.0,
                help="Interpretación: unidades upsell (bebidas+complementos) por ticket.",
            ) / 100.0

        with cB:
            top_n = st.number_input(
                "Top N",
                min_value=3, max_value=30, value=10, step=1,
                help="Cuántos conceptos mostrar en el top.",
            )

        with cC:
            mensaje_meta = st.text_area(
                "Texto / meta (se incluye en el correo)",
                value="Meta: elevar attach de bebidas+complementos con sugerencia activa en caja y combos.",
                height=90,
            )

        # GAP vs meta (unidades)
        unidades_meta_sem = tickets_sem * attach_meta
        gap_unidades_sem = max(unidades_meta_sem - unidades_sem, 0.0)

        unidades_meta_4 = tickets_4 * attach_meta
        gap_unidades_4 = max(unidades_meta_4 - unidades_4, 0.0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Unidades upsell (Semana)", f"{unidades_sem:,.0f}")
        k2.metric("Attach real (Semana)", fmt_pct(attach_real_sem))
        k3.metric("Gap unidades (Semana)", f"{gap_unidades_sem:,.0f}")
        k4.metric("Meta attach", fmt_pct(attach_meta))

        kk1, kk2, kk3, kk4 = st.columns(4)
        kk1.metric("Unidades upsell (4 Sem)", f"{unidades_4:,.0f}")
        kk2.metric("Attach real (4 Sem)", fmt_pct(attach_real_4))
        kk3.metric("Gap unidades (4 Sem)", f"{gap_unidades_4:,.0f}")
        kk4.metric("Tickets (Semana)", f"{int(tickets_sem):,}")

        # Tablas top
        st.markdown("#### Top upsells por unidades (real)")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Semana actual**")
            if ups_sem.empty:
                st.info("No se detectaron upsells mapeados en la semana actual.")
            else:
                st.dataframe(ups_sem.head(int(top_n)), use_container_width=True)

        with t2:
            st.markdown("**4 semanas actuales**")
            if ups_4.empty:
                st.info("No se detectaron upsells mapeados en las 4 semanas actuales.")
            else:
                st.dataframe(ups_4.head(int(top_n)), use_container_width=True)

        # Correo via Apps Script (solo envío)
        st.markdown("---")
        st.markdown("## Enviar correo")

        if not APPSCRIPT_URL:
            st.info("Configura `APPSCRIPT_URL` en `.streamlit/secrets.toml` para habilitar el envío.")
        else:
            # Vista previa del correo (lo que mandaremos)
            payload = {
                "restaurante": rest_seleccionado,
                "periodos": {
                    "semana_actual": f"{semana_actual_inicio.strftime('%d/%m/%Y')} - {semana_actual_fin.strftime('%d/%m/%Y')}",
                    "cuatro_sem_actual": f"{cuatro_sem_actual_inicio.strftime('%d/%m/%Y')} - {cuatro_sem_actual_fin.strftime('%d/%m/%Y')}",
                },
                "meta": {
                    "attach_meta": float(attach_meta),
                    "mensaje": mensaje_meta,
                },
                "resultados": {
                    "tickets_sem": int(tickets_sem),
                    "unidades_sem": float(unidades_sem),
                    "attach_real_sem": float(attach_real_sem),
                    "gap_unidades_sem": float(gap_unidades_sem),

                    "tickets_4": int(tickets_4),
                    "unidades_4": float(unidades_4),
                    "attach_real_4": float(attach_real_4),
                    "gap_unidades_4": float(gap_unidades_4),
                },
                "top_sem": ups_sem.head(int(top_n)).to_dict(orient="records") if not ups_sem.empty else [],
                "top_4": ups_4.head(int(top_n)).to_dict(orient="records") if not ups_4.empty else [],
            }


            if st.button("📩 Enviar reporte"):
                try:
                    resp = post_json(APPSCRIPT_URL, payload)
                    if resp.get("ok"):
                        st.success(f"Correo enviado a: {resp.get('to', '—')}")
                    else:
                        st.error(f"Apps Script error: {resp.get('error')}")
                except Exception as e:
                    st.error(f"No se pudo contactar Apps Script: {e}")
