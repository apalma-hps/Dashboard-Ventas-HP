# pages/03_Week_over_Week.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

st.sidebar.markdown("### Actualización")

if st.sidebar.button("🔄 Actualizar data"):
    st.cache_data.clear()   # limpia caché de load_data / load_catalogo (y demás cache_data)
    st.rerun()              # vuelve a ejecutar la app

st.sidebar.caption(f"Última vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# ============= CONFIG BÁSICA =============
st.set_page_config(
    page_title="Week over Week – Marcas HP",
    page_icon="📈",
    layout="wide",
)

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
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.10);
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
DATA_URL= "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
#DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQZBL6mvIC1OUC-p0MREMW_7UvMKb8It4Y_ldFOi3FbqP4cwZBLrDXwpA_hjBzkeZz3tsOBqd9BlamY/pub?output=csv"

COL_CC = "Restaurante"
COL_ESTADO = "Estado"
COL_FECHA = "Fecha"
COL_TOTAL = "Total"
COL_TIPO = "Tipo"
COL_FOLIO = "Folio"
COL_VENTAS = "ventas_efectivas"

def fmt_money(x):
    return "—" if (x is None or pd.isna(x)) else f"${x:,.0f}"

def fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{x * 100:,.1f}%"

def fmt_pp(x):
    # puntos porcentuales (pp) para cambios en proporciones
    return "—" if (x is None or pd.isna(x)) else f"{x * 100:+.1f}pp"

def fmt_change_ratio(x):
    # cambio porcentual: 0.12 => +12.0%
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
    # (cur-prev)/prev con manejo de 0
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

# =========================================================
# MÉTRICAS + COLOR DELTAS (REEMPLAZA ESTAS FUNCIONES)
# =========================================================

def calcular_metricas(df_periodo: pd.DataFrame):
    """
    KPIs:
    - ventas: suma ventas_efectivas (dinero)
    - tickets: folios únicos donde ventas_efectivas > 0
    - cancelados: folios únicos donde Estado=void
    - ticket_promedio: ventas / tickets
    - ventas_delivery: suma ventas_efectivas donde Tipo contiene delivery (dinero)
    - pct_delivery: ventas_delivery / ventas  (dinero/dinero) ✅
    - orders_per_day: tickets / días únicos (ENTERO) ✅
    """
    if df_periodo is None or df_periodo.empty:
        return {
            "ventas": 0.0,
            "tickets": 0,
            "cancelados": 0,
            "ticket_promedio": 0.0,
            "ventas_delivery": 0.0,
            "pct_delivery": 0.0,
            "orders_per_day": 0,  # ✅ entero
        }

    ventas = float(df_periodo[COL_VENTAS].sum())

    tickets_validos = df_periodo[df_periodo[COL_VENTAS] > 0]
    tickets = int(tickets_validos[COL_FOLIO].nunique()) if COL_FOLIO in tickets_validos.columns else 0

    cancelados = 0
    if (COL_ESTADO in df_periodo.columns) and (COL_FOLIO in df_periodo.columns):
        estado_norm = df_periodo[COL_ESTADO].astype(str).str.strip().str.lower()
        cancelados = int(df_periodo.loc[estado_norm.eq("void"), COL_FOLIO].nunique())

    ticket_promedio = float(ventas / tickets) if tickets > 0 else 0.0

    # ✅ ventas_delivery en dinero (ventas_efectivas) para que %delivery sea dinero/dinero
    if COL_TIPO in df_periodo.columns:
        ventas_delivery = float(df_periodo.loc[df_periodo[COL_TIPO].map(is_delivery), COL_VENTAS].sum())
    else:
        ventas_delivery = 0.0

    pct_delivery = float(ventas_delivery / ventas) if ventas > 0 else 0.0

    dias_unicos = int(df_periodo[COL_FECHA].dt.date.nunique())
    orders_per_day = int(round(tickets / dias_unicos)) if dias_unicos > 0 else 0  # ✅ entero

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
    """
    Streamlit st.metric:
    - normal  => positivo verde, negativo rojo
    - inverse => positivo rojo, negativo verde
    - off     => sin delta / sin base

    Regla:
    - invert=False para KPIs "buenos" (sube=verde)
    - invert=True  para KPIs "malos"  (sube=rojo) -> Cancelados
    """
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

# Validaciones mínimas
required_cols = {COL_CC, COL_FECHA, COL_TOTAL}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en la base: {missing}")
    st.stop()

df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
df[COL_TOTAL] = clean_money_series(df[COL_TOTAL])

# Excluir void => ventas_efectivas=0
if COL_ESTADO in df.columns:
    estado_norm = df[COL_ESTADO].astype(str).str.strip().str.lower()
    is_void = estado_norm.eq("void")
else:
    is_void = pd.Series(False, index=df.index)

df[COL_VENTAS] = np.where(is_void, 0.0, df[COL_TOTAL].fillna(0.0))

# Filtrar fechas válidas
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
# Semana actual (seleccionada)
semana_actual_inicio = inicio_semana_sel
semana_actual_fin = semana_actual_inicio + timedelta(days=6)

# Semana anterior
semana_anterior_inicio = semana_actual_inicio - timedelta(days=7)
semana_anterior_fin = semana_anterior_inicio + timedelta(days=6)

# 4 semanas actuales: 28 días terminando en el domingo de la semana seleccionada
cuatro_sem_actual_inicio = semana_actual_inicio - timedelta(days=21)  # lunes de 3 semanas antes
cuatro_sem_actual_fin = semana_actual_fin

# 4 semanas anteriores: 28 días previos al inicio de las 4 semanas actuales
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
# KPIs WoW (REEMPLAZA TU BLOQUE COMPLETO)
# =========================================================
st.markdown("### Week over Week (WoW)")

cambio_ventas_wow = safe_pct_change(metricas_sem_actual["ventas"], metricas_sem_anterior["ventas"])
cambio_tickets_wow = safe_pct_change(metricas_sem_actual["tickets"], metricas_sem_anterior["tickets"])
cambio_ticket_prom_wow = safe_pct_change(metricas_sem_actual["ticket_promedio"], metricas_sem_anterior["ticket_promedio"])
cambio_orders_day_wow = safe_pct_change(metricas_sem_actual["orders_per_day"], metricas_sem_anterior["orders_per_day"])
cambio_cancelados_wow = safe_pct_change(metricas_sem_actual["cancelados"], metricas_sem_anterior["cancelados"])

# Delivery:
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
        delta_color=delta_color(cambio_cancelados_wow, invert=True),  # ✅ al revés
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
        f"{metricas_sem_actual['orders_per_day']:,.0f}",  # ✅ entero
        delta=f"{fmt_change_ratio(cambio_orders_day_wow)} vs sem. anterior",
        delta_color=delta_color(cambio_orders_day_wow, invert=False),
    )

with col6:
    st.metric(
        "% Delivery",
        fmt_pct(metricas_sem_actual["pct_delivery"]),
        delta=f"{fmt_pp(cambio_pct_delivery_wow_pp)} vs sem. anterior",
        delta_color=delta_color(cambio_pct_delivery_wow_pp, invert=False),  # ✅ ya no gris
    )


st.markdown("#### Comparativa Detallada WoW")
wow_data = pd.DataFrame({
    "Métrica": ["Ventas", "Tickets", "Cancelados", "Ticket Promedio", "Órdenes / día", "Ventas Delivery", "% Delivery"],
    "Semana Anterior": [
        fmt_money(metricas_sem_anterior["ventas"]),
        f"{metricas_sem_anterior['tickets']:,.0f}",
        f"{metricas_sem_anterior['cancelados']:,.0f}",
        fmt_money(metricas_sem_anterior["ticket_promedio"]),
        f"{metricas_sem_anterior['orders_per_day']:,.0f}",  # ✅ entero
        fmt_money(metricas_sem_anterior["ventas_delivery"]),
        fmt_pct(metricas_sem_anterior["pct_delivery"]),
    ],
    "Semana Actual": [
        fmt_money(metricas_sem_actual["ventas"]),
        f"{metricas_sem_actual['tickets']:,.0f}",
        f"{metricas_sem_actual['cancelados']:,.0f}",
        fmt_money(metricas_sem_actual["ticket_promedio"]),
        f"{metricas_sem_actual['orders_per_day']:,.0f}",  # ✅ entero
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
# KPIs 4WoW (REEMPLAZA TU BLOQUE COMPLETO)
# =========================================================
st.markdown("### 4 Weeks vs 4 Weeks (4WoW)")

cambio_ventas_4wow = safe_pct_change(metricas_4sem_actual["ventas"], metricas_4sem_anterior["ventas"])
cambio_tickets_4wow = safe_pct_change(metricas_4sem_actual["tickets"], metricas_4sem_anterior["tickets"])
cambio_ticket_prom_4wow = safe_pct_change(metricas_4sem_actual["ticket_promedio"], metricas_4sem_anterior["ticket_promedio"])
cambio_orders_day_4wow = safe_pct_change(metricas_4sem_actual["orders_per_day"], metricas_4sem_anterior["orders_per_day"])

# Delivery:
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
        f"{metricas_4sem_actual['orders_per_day']:,.0f}",  # ✅ entero
        delta=f"{fmt_change_ratio(cambio_orders_day_4wow)} vs 4 sem. anteriores",
        delta_color=delta_color(cambio_orders_day_4wow, invert=False),
    )

with col5:
    st.metric(
        "% Delivery",
        fmt_pct(metricas_4sem_actual["pct_delivery"]),
        delta=f"{fmt_pp(cambio_pct_delivery_4wow_pp)} vs 4 sem. anteriores",
        delta_color=delta_color(cambio_pct_delivery_4wow_pp, invert=False),  # ✅ ya no gris
    )

st.markdown("#### Comparativa Detallada 4WoW")
four_wow_data = pd.DataFrame({
    "Métrica": ["Ventas", "Tickets", "Ticket Promedio", "Órdenes / día", "Ventas Delivery", "% Delivery"],
    "4 Semanas Anteriores": [
        fmt_money(metricas_4sem_anterior["ventas"]),
        f"{metricas_4sem_anterior['tickets']:,.0f}",
        fmt_money(metricas_4sem_anterior["ticket_promedio"]),
        f"{metricas_4sem_anterior['orders_per_day']:,.0f}",  # ✅ entero
        fmt_money(metricas_4sem_anterior["ventas_delivery"]),
        fmt_pct(metricas_4sem_anterior["pct_delivery"]),
    ],
    "4 Semanas Actuales": [
        fmt_money(metricas_4sem_actual["ventas"]),
        f"{metricas_4sem_actual['tickets']:,.0f}",
        fmt_money(metricas_4sem_actual["ticket_promedio"]),
        f"{metricas_4sem_actual['orders_per_day']:,.0f}",  # ✅ entero
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
# TENDENCIA SEMANAL (últimas 8 semanas desde semana seleccionada)
# =========================================================
st.markdown("### Tendencia de Ventas Semanales (últimas 8 semanas)")

fecha_8_semanas_atras = semana_actual_inicio - timedelta(days=7 * 7)  # 7 semanas antes
df_ultimas_8_sem = df[(df[COL_FECHA] >= fecha_8_semanas_atras) & (df[COL_FECHA] <= semana_actual_fin)].copy()

if rest_seleccionado != "Todos los restaurantes":
    df_ultimas_8_sem = df_ultimas_8_sem[df_ultimas_8_sem[COL_CC] == rest_seleccionado]

if df_ultimas_8_sem.empty:
    st.info("No hay datos suficientes para mostrar la tendencia semanal.")
else:
    df_ultimas_8_sem["semana"] = df_ultimas_8_sem[COL_FECHA].apply(lambda x: to_monday(x))
    ventas_semanales = (
        df_ultimas_8_sem.groupby("semana", as_index=False)[COL_VENTAS]
        .sum()
        .sort_values("semana")
    )

    chart = (
        alt.Chart(ventas_semanales)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("semana:T", title="Semana (inicio)"),
            y=alt.Y(f"{COL_VENTAS}:Q", title="Ventas"),
            tooltip=[
                alt.Tooltip("semana:T", title="Semana", format="%d/%m/%Y"),
                alt.Tooltip(f"{COL_VENTAS}:Q", title="Ventas", format="$,.0f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

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
        # WoW
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

        # 4WoW
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

    # Ordenar por ventas actuales (WoW) desc
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

    # Mini chart: cambio ventas % WoW
    st.markdown("#### Cambios % de Ventas (WoW) · ranking")
    df_chart = df_comp_wow[["Restaurante", "Cambio Ventas %"]].copy()
    df_chart["Cambio Ventas %"] = df_chart["Cambio Ventas %"].fillna(0.0)

    ch_bar = (
        alt.Chart(df_chart)
        .mark_bar()
        .encode(
            x=alt.X("Cambio Ventas %:Q", title="Cambio % Ventas (WoW)", axis=alt.Axis(format="%")),
            y=alt.Y("Restaurante:N", sort="-x", title=""),
            tooltip=[alt.Tooltip("Restaurante:N"), alt.Tooltip("Cambio Ventas %:Q", format=".1%")],
        )
        .properties(height=min(700, 24 * max(8, len(df_chart))))
    )
    st.altair_chart(ch_bar, use_container_width=True)

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

    st.markdown("#### Cambios % de Ventas (4WoW) · ranking")
    df_chart2 = df_comp_4wow[["Restaurante", "Cambio Ventas %"]].copy()
    df_chart2["Cambio Ventas %"] = df_chart2["Cambio Ventas %"].fillna(0.0)

    ch_bar2 = (
        alt.Chart(df_chart2)
        .mark_bar()
        .encode(
            x=alt.X("Cambio Ventas %:Q", title="Cambio % Ventas (4WoW)", axis=alt.Axis(format="%")),
            y=alt.Y("Restaurante:N", sort="-x", title=""),
            tooltip=[alt.Tooltip("Restaurante:N"), alt.Tooltip("Cambio Ventas %:Q", format=".1%")],
        )
        .properties(height=min(700, 24 * max(8, len(df_chart2))))
    )
    st.altair_chart(ch_bar2, use_container_width=True)

