# Inicio.py — Dashboard Ventas BYF W&C (Streamlit + Google Sheets)
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import re
import altair as alt




# ===== Tema de Altair (paleta profesional aqua/teal) =====


def byf_altair_theme():
    return {
        "config": {
            "background": "rgba(0,0,0,0)",  # fondo transparente
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": "#1B1D22",
                "titleColor": "#1B1D22",
            },
            "legend": {
                "labelColor": "#1B1D22",
                "titleColor": "#1B1D22",
            },
            "range": {
                "category": [
                    "#1B1D22",  # Graphite Ink
                    "#7AD9CF",  # Teal Mist
                    "#A7F0E3",  # Aqua Glow
                    "#B8EDEA",  # Ice Gradient Blue
                    "#6F7277",  # Soft Grey
                    "#37D2A3",  # Verde éxito suave
                ],
            },
        }
    }

alt.themes.register("byf_theme", byf_altair_theme)
alt.themes.enable("byf_theme")
# ===== Fin tema Altair =====



st.set_page_config(page_title="Ops – Ventas", page_icon="📊", layout="wide")

st.sidebar.markdown("### Actualización")

if st.sidebar.button("🔄 Actualizar data"):
    st.cache_data.clear()   # limpia caché de load_data / load_catalogo (y demás cache_data)
    st.rerun()              # vuelve a ejecutar la app

st.sidebar.caption(f"Última vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
# ===== Estilos personalizados tipo "poster" (glassmorphism) =====



st.markdown(
    """
    <style>
    /* Fondo general con degradados suaves aqua */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 80% 10%, #A7F0E3 0, #F5F8F9 40%),
                    radial-gradient(circle at 0% 80%, #B8EDEA 0, #F5F8F9 45%),
                    #F5F8F9;
    }

    /* Quitar fondo sólido del header superior */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Sidebar estilo tarjeta translúcida */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(14px);
        border-right: 1px solid rgba(148, 163, 184, 0.35);
    }

    /* Centrar y limitar ancho del contenido principal */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Métricas en formato "card" */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.90);
        border-radius: 18px;
        padding: 1rem 1.3rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    div[data-testid="stMetricLabel"] {
        color: #6F7277;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stMetricValue"] {
        color: #1B1D22;
        font-weight: 600;
        font-size: 1.5rem;
    }

    /* Tabs más limpias */
    div[data-testid="stTabs"] button[role="tab"] {
        border-radius: 999px;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        border: 1px solid transparent;
        background-color: rgba(255,255,255,0.6);
        color: #6F7277;
        font-weight: 500;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: #7AD9CF;
        color: #1B1D22;
        border-color: rgba(148, 163, 184, 0.5);
    }

    /* Dataframes con bordes suaves */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 0.3rem 0.3rem 0.8rem 0.3rem;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    /* Centrar completamente las métricas */
    div[data-testid="stMetric"] {
        display: flex;
        flex-direction: column;
        justify-content: center;  /* centra vertical */
        align-items: center;      /* centra horizontal */
        text-align: center;       /* centra textos */
    
    img {
        border-radius: 50%;
        border: 4px solid #A7F0E3;
        box-shadow: 0 12px 30px rgba(125, 211, 222, 0.45);
    }

}
    
    </style>
    """,
    unsafe_allow_html=True,
)
# ===== Fin estilos personalizados =====


# URL pública de Google Sheets (CSV)
DATA_URL= "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
#DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQZBL6mvIC1OUC-p0MREMW_7UvMKb8It4Y_ldFOi3FbqP4cwZBLrDXwpA_hjBzkeZz3tsOBqd9BlamY/pub?output=csv"


# ---------- Utilidades ----------
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df.columns = [c.strip() for c in df.columns]
    return df


def fmt_money(x):
    return "—" if pd.isna(x) else f"${x:,.0f}"


def fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{x * 100:,.0f}%"


def arrow_gap(p):
    if p is None or pd.isna(p):
        return "●"  # gris (sin dato/base)
    if p > 0:
        return "▲"
    if p < 0:
        return "▼"
    return "●"  # 0 → punto gris


def color_dot_for_target(p):
    # Verde >=100%, Amarillo 80–99%, Rojo <80%, Gris sin dato
    if p is None or pd.isna(p):
        return "⚪"
    if p >= 1.0:
        return "🟢"
    if p >= 0.80:
        return "🟡"
    return "🔴"


def is_delivery(val):
    try:
        return "delivery" in str(val).lower()
    except Exception:
        return False


def agregar_periodo(df_src: pd.DataFrame, gran: str, col_fecha: str) -> pd.DataFrame:
    g = df_src.copy()
    if gran == "Día":
        g["periodo"] = g[col_fecha].dt.to_period("D").dt.to_timestamp()
    elif gran == "Semana":
        # Semana que inicia en lunes; usamos el inicio de la semana como fecha
        g["periodo"] = g[col_fecha].dt.to_period("W-MON").apply(lambda r: r.start_time)
    else:  # "Mes"
        g["periodo"] = g[col_fecha].dt.to_period("M").dt.to_timestamp()
    return g


# ---------- Carga de datos ----------
df = load_data()

# Columnas base (usa EXACTAMENTE estos nombres)
COL_CC = "Restaurante"     # CC / Marca
COL_FECHA = "Fecha"        # fecha de la venta
COL_TOTAL = "Total"
COL_ESTADO= "Estado"# ventas netas
COL_TIPO = "Tipo"          # detectar delivery
COL_FOLIO = "Folio"        # tickets
COL_DETALLE = "Detalle Items"  # para top productos / complementos
COL_VENTAS = COL_TOTAL

# Tipificación mínima
df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)


df[COL_TOTAL] = (
    df[COL_TOTAL]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

df = df[
    df[COL_ESTADO]
    .astype(str)
    .str.strip()
    .str.lower()
    .ne("void")
].copy()

# ===== Header con Logo + Título Moderno =====
LOGO_URL = "https://raw.githubusercontent.com/apalma-hps/Dashboard-Ventas-HP/main/logo_hp.png"


st.markdown(
    f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        margin: 1.5rem 0 2.5rem 0;
    ">
        <div style="
            width: 88px;
            height: 88px;
            border-radius: 999px;
            border: 3px solid #7AD9CF;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
        ">
            <img src="{LOGO_URL}" style="
                width: 70px;
                height: 70px;
                border-radius: 999px;
                object-fit: cover;
            ">
        </div>
        <div style="text-align: left;">
            <div style="
                font-size: 1.9rem;
                font-weight: 700;
                color: #1B1D22;
            ">
                Ventas y Health Rate – Marcas HP
            </div>
            <div style="
                font-size: 0.95rem;
                color: #6F7277;
                margin-top: 4px;
            ">
                 Rendimiento mensual.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ===== Fin header =====



# =====================================================
# HEALTH RATE GLOBAL (NO RESPETA FILTROS)
# =====================================================
st.markdown("### Resumen anual")

df_all = df.dropna(subset=[COL_FECHA, COL_VENTAS]).copy()

if df_all.empty:
    st.info("No hay datos para calcular el Health Rate global.")
else:
    # ---------- KPIs ANUALES / TODA LA DATA ----------
    ventas_totales_all = df_all[COL_VENTAS].sum()

    # Folios únicos en toda la base
    n_tickets_all = (
        df_all[COL_FOLIO].nunique()
        if COL_FOLIO in df_all.columns
        else None
    )

    ticket_prom_all = (
        ventas_totales_all / n_tickets_all
        if n_tickets_all and n_tickets_all > 0
        else None
    )

    # Órdenes promedio por día (en todo el periodo disponible)
    dias_unicos_all = df_all[COL_FECHA].dt.date.nunique()
    orders_per_day_all = (
        n_tickets_all / dias_unicos_all
        if (n_tickets_all and dias_unicos_all)
        else None
    )

    # % Delivery sobre todas las ventas
    pct_delivery_all = None
    if COL_TIPO in df_all.columns and ventas_totales_all:
        tot_del_all = df_all.loc[
            df_all[COL_TIPO].map(is_delivery), COL_VENTAS
        ].sum()
        pct_delivery_all = (
            tot_del_all / ventas_totales_all if ventas_totales_all else None
        )

    cG1, cG2, cG3, cG4 = st.columns(4)
    cG1.metric("Ventas Netas  Anuales", fmt_money(ventas_totales_all))
    cG2.metric(
        "Ticket Promedio",
        fmt_money(ticket_prom_all) if ticket_prom_all is not None else "—",
    )
    cG3.metric(
        "Promedio de Órdenes Diarias",
        f"{orders_per_day_all:,.0f}" if orders_per_day_all is not None else "—",
    )

    cG4.metric(
        "Aportación Delivery",
        fmt_pct(pct_delivery_all),
    )

    # A partir de aquí puedes seguir con tus gráficas globales
    df_all["mes"] = df_all[COL_FECHA].dt.to_period("M").dt.to_timestamp()
    # (y luego la gráfica por restaurante global que ya teníamos)

    # ---------- Gráfica 2: ventas mensuales por restaurante (top) ----------
    df_all_rest = df_all.dropna(subset=[COL_CC]).copy()

    tot_por_rest_global = (
        df_all_rest.groupby(COL_CC, as_index=False)[COL_VENTAS]
        .sum()
        .sort_values(COL_VENTAS, ascending=False)
    )
    top_rest_global = tot_por_rest_global[COL_CC].head(8).tolist()

    df_rest_mes_global = (
        df_all_rest[df_all_rest[COL_CC].isin(top_rest_global)]
        .groupby(["mes", COL_CC], as_index=False)[COL_VENTAS]
        .sum()
        .sort_values(["mes", COL_CC])
    )

    st.markdown("#### Ventas Mensuales por Marca")
    ch_global_rest = (
        alt.Chart(df_rest_mes_global)
        .mark_line(point=True)
        .encode(
            x=alt.X("mes:T", title="Mes"),
            y=alt.Y(f"{COL_VENTAS}:Q", title="Ventas netas"),
            color=alt.Color(f"{COL_CC}:N", title="Restaurante"),
            tooltip=[
                alt.Tooltip("mes:T", title="Mes"),
                alt.Tooltip(f"{COL_CC}:N", title="Restaurante"),
                alt.Tooltip(f"{COL_VENTAS}:Q", title="Ventas", format=","),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(ch_global_rest, use_container_width=True)

st.markdown("---")

# =====================================================
# TABLA: TICKET PROMEDIO POR RESTAURANTE (GLOBAL)
# =====================================================
st.markdown("#### Ticket Promedio por Restaurante ")

if COL_FOLIO not in df_all.columns:
    st.info("No existe la columna de folios para calcular ticket promedio por restaurante.")
else:
    df_tickets = df_all.dropna(subset=[COL_CC, COL_FOLIO]).copy()

    if df_tickets.empty:
        st.info("No hay datos para calcular el ticket promedio por restaurante.")
    else:
        resumen_ticket = (
            df_tickets.groupby(COL_CC, as_index=False)
            .agg(
                ventas=(COL_VENTAS, "sum"),
                tickets=(COL_FOLIO, "nunique"),
            )
        )
        resumen_ticket["ticket_promedio"] = resumen_ticket.apply(
            lambda row: row["ventas"] / row["tickets"] if row["tickets"] > 0 else None,
            axis=1,
        )

        resumen_ticket = resumen_ticket.sort_values("ventas", ascending=False)

        resumen_view = resumen_ticket[["Restaurante", "ventas", "tickets", "ticket_promedio"]]
        resumen_view = resumen_view.rename(
            columns={
                "ventas": "Ventas Netas",
                "tickets": "Tickets",
                "ticket_promedio": "Ticket Promedio",
            }
        )

        # 🔥 Quitar los números de fila
        resumen_view.index = [''] * len(resumen_view)

        st.dataframe(
            resumen_view.style.format(
                {
                    "Ventas Netas": "${:,.0f}",
                    "Tickets": "{:,.0f}",
                    "Ticket Promedio": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )

