# pages/05_Ticket_Promedio.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

# ============= CONFIG BÁSICA =============
st.set_page_config(
    page_title="Ticket Promedio – Marcas HP",
    page_icon="🧾",
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
        max-width: 1200px;
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
        <h1 style="margin-bottom:0;">Ticket Promedio – Marcas HP</h1>
        <p style="color:#6F7277;font-size:0.95rem;margin-top:0.25rem;">
        Ticket promedio por Día/Semana/Mes + variaciones, delivery y tendencias.
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================================
# URL
# =========================================================
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"

# =========================================================
# Helpers
# =========================================================
def fmt_money(x):
    return "—" if (x is None or pd.isna(x)) else f"${x:,.0f}"

def fmt_pct_pp(x):
    return "—" if (x is None or pd.isna(x)) else f"{x * 100:,.1f}%"

def clean_money_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

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

def agregar_periodo(df_src: pd.DataFrame, gran: str, col_fecha: str) -> pd.DataFrame:
    g = df_src.copy()
    if gran == "Día":
        g["periodo"] = g[col_fecha].dt.to_period("D").dt.to_timestamp()
    elif gran == "Semana":
        g["periodo"] = g[col_fecha].dt.to_period("W-MON").apply(lambda r: r.start_time)
    else:  # Mes
        g["periodo"] = g[col_fecha].dt.to_period("M").dt.to_timestamp()
    return g

def periodo_label(serie_periodo: pd.Series, gran: str) -> pd.Series:
    p = pd.to_datetime(serie_periodo)
    if gran == "Día":
        return p.dt.strftime("%d %b %Y")
    if gran == "Semana":
        return p.dt.strftime("Sem %d %b %Y")
    return p.dt.strftime("%b %Y")

def is_delivery_tipo(val) -> bool:
    try:
        s = str(val).strip().lower()
    except Exception:
        return False

    delivery_set = {
        "delivery",
        "delivery (delivery)",
        "takeout",
        "takeout (delivery)",
    }
    return s in delivery_set

# =========================================================
# Carga de datos
# =========================================================
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_URL)
    df_.columns = [c.strip() for c in df_.columns]
    return df_

df = load_data()

# =========================================================
# Columnas esperadas
# =========================================================
COL_CC = "Restaurante"
COL_ESTADO = "Estado"
COL_FECHA = "Fecha"
COL_SUBTOT = "Subtotal"
COL_TOTAL = "Total"
COL_DESCUENTOS = "Descuentos"
COL_FOLIO = "Folio"
COL_VENTAS = "ventas_efectivas"
COL_TIPO = "Tipo"

missing = [c for c in [COL_CC, COL_FECHA, COL_TOTAL] if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en la base: {missing}")
    st.stop()

df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
df[COL_TOTAL] = clean_money_series(df[COL_TOTAL]).fillna(0.0)

if COL_SUBTOT not in df.columns:
    df[COL_SUBTOT] = df[COL_TOTAL]
else:
    df[COL_SUBTOT] = clean_money_series(df[COL_SUBTOT]).fillna(0.0)

if COL_DESCUENTOS not in df.columns:
    df[COL_DESCUENTOS] = 0.0
else:
    df[COL_DESCUENTOS] = clean_money_series(df[COL_DESCUENTOS]).fillna(0.0)

tax_col = detect_tax_column(df)
if tax_col is not None:
    df["_impuestos"] = clean_money_series(df[tax_col]).fillna(0.0)
else:
    df["_impuestos"] = (df[COL_TOTAL] - df[COL_SUBTOT]).clip(lower=0.0)

is_void = get_void_mask(df, COL_ESTADO)
df["_calc_sti_d"] = (df[COL_SUBTOT] + df["_impuestos"] - df[COL_DESCUENTOS]).fillna(0.0)
df["_ventas_brutas_regla"] = np.where(df["_calc_sti_d"] > df[COL_TOTAL], df[COL_TOTAL], df["_calc_sti_d"])
df[COL_VENTAS] = np.where(is_void, 0.0, pd.Series(df["_ventas_brutas_regla"], index=df.index).clip(lower=0.0))

# =========================================================
# Sidebar filtros
# =========================================================
st.sidebar.markdown("### Filtros")

if df[COL_FECHA].notna().any():
    min_f, max_f = df[COL_FECHA].min(), df[COL_FECHA].max()
    rango = st.sidebar.date_input("Rango de fechas", value=(min_f.date(), max_f.date()))
    if isinstance(rango, (list, tuple)) and len(rango) == 2:
        f_ini, f_fin = [pd.to_datetime(x) for x in rango]
    else:
        f_ini = f_fin = pd.to_datetime(rango[0]) if isinstance(rango, (list, tuple)) else pd.to_datetime(rango)
else:
    st.error("No hay fechas válidas en la base.")
    st.stop()

rests_all = sorted(df[COL_CC].dropna().unique().tolist())
sel_rests = st.sidebar.multiselect("Restaurantes", options=rests_all, default=rests_all)

gran = st.sidebar.radio("Granularidad del ticket", ["Día", "Semana", "Mes"], index=2, horizontal=True)

mostrar_rolling = st.sidebar.checkbox("Mostrar promedio móvil (rolling)", value=True)
rolling_n = st.sidebar.select_slider(
    "Ventana rolling (periodos)",
    options=[3, 4, 5, 7, 8, 12],
    value=4
)

# =========================================================
# Filtrar
# =========================================================
df_f = df.copy()
mask_date = (df_f[COL_FECHA].dt.date >= f_ini.date()) & (df_f[COL_FECHA].dt.date <= f_fin.date())
df_f = df_f[mask_date]

if sel_rests:
    df_f = df_f[df_f[COL_CC].isin(sel_rests)]

if df_f.empty:
    st.info("No hay datos con los filtros seleccionados.")
    st.stop()

# =========================================================
# Tickets
# =========================================================
if COL_FOLIO not in df_f.columns:
    st.warning("No existe columna 'Folio'. Fallback: tickets = conteo de filas NO VOID (puede inflar tickets).")

def tickets_no_void(g: pd.DataFrame) -> int:
    mvoid = get_void_mask(g, COL_ESTADO)
    if COL_FOLIO in g.columns:
        return int(g.loc[~mvoid, COL_FOLIO].nunique())
    return int((~mvoid).sum())

# =========================================================
# Serie por periodo
# =========================================================
df_p = agregar_periodo(df_f, gran, COL_FECHA)

def safe_tipo_series(g: pd.DataFrame) -> pd.Series:
    # Si no existe COL_TIPO en g, devolvemos una serie vacía alineada al index
    if COL_TIPO in g.columns:
        return g[COL_TIPO]
    return pd.Series("", index=g.index)

serie = (
    df_p.groupby("periodo", as_index=False)
    .apply(lambda g: pd.Series({
        "ventas_totales": float(g[COL_VENTAS].sum()),
        "tickets_totales": tickets_no_void(g),

        "ventas_delivery": float(
            g.loc[safe_tipo_series(g).map(is_delivery_tipo), COL_VENTAS].sum()
        ),
        "tickets_delivery": tickets_no_void(
            g.loc[safe_tipo_series(g).map(is_delivery_tipo)]
        ),
    }))
    .reset_index(drop=True)
    .sort_values("periodo")
)

# Labels + orden
serie["periodo_str"] = periodo_label(serie["periodo"], gran)
orden_periodos = (
    serie[["periodo", "periodo_str"]]
    .drop_duplicates()
    .sort_values("periodo")["periodo_str"]
    .tolist()
)

# Ticket promedio total y delivery
serie["ticket_promedio"] = np.where(
    serie["tickets_totales"] > 0,
    serie["ventas_totales"] / serie["tickets_totales"],
    np.nan
)

serie["ticket_promedio_delivery"] = np.where(
    serie["tickets_delivery"] > 0,
    serie["ventas_delivery"] / serie["tickets_delivery"],
    np.nan
)

# Variación vs periodo anterior (TOTAL)
serie["tp_prev"] = serie["ticket_promedio"].shift(1)
serie["var_abs"] = serie["ticket_promedio"] - serie["tp_prev"]
serie["var_pct"] = np.where(
    serie["tp_prev"].notna() & (serie["tp_prev"] != 0),
    serie["var_abs"] / serie["tp_prev"],
    np.nan
)

# Rolling
if mostrar_rolling:
    serie["tp_roll"] = serie["ticket_promedio"].rolling(
        rolling_n, min_periods=max(2, rolling_n // 2)
    ).mean()

# =========================================================
# KPIs globales
# =========================================================
ventas_tot = float(df_f[COL_VENTAS].sum())
tickets_tot = tickets_no_void(df_f)
tp_global = (ventas_tot / tickets_tot) if tickets_tot > 0 else np.nan

last_valid = serie.dropna(subset=["ticket_promedio"]).copy()
ult_tp = float(last_valid["ticket_promedio"].iloc[-1]) if not last_valid.empty else np.nan
ult_var_pct = float(last_valid["var_pct"].iloc[-1]) if (not last_valid.empty and pd.notna(last_valid["var_pct"].iloc[-1])) else np.nan
ult_var_abs = float(last_valid["var_abs"].iloc[-1]) if (not last_valid.empty and pd.notna(last_valid["var_abs"].iloc[-1])) else np.nan

best_row = last_valid.loc[last_valid["ticket_promedio"].idxmax()] if not last_valid.empty else None
worst_row = last_valid.loc[last_valid["ticket_promedio"].idxmin()] if not last_valid.empty else None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ticket promedio (global)", fmt_money(tp_global))
k2.metric("Ventas (global)", fmt_money(ventas_tot))
k3.metric("Tickets (global)", f"{tickets_tot:,}")

if pd.notna(ult_tp):
    delta_txt = f"{ult_var_abs:,.0f}" if pd.notna(ult_var_abs) else None
    k4.metric(f"Último {gran.lower()} (ticket prom.)", fmt_money(ult_tp), delta=delta_txt)
    if pd.notna(ult_var_pct):
        st.caption(f"Variación vs periodo anterior: {fmt_pct_pp(ult_var_pct)}")
else:
    k4.metric(f"Último {gran.lower()} (ticket prom.)", "—")

st.markdown("---")

# =========================================================
# Gráfica principal
# =========================================================
st.markdown("### Ticket promedio en el tiempo")

base = alt.Chart(serie)

line = base.mark_line(point=True).encode(
    x=alt.X("periodo_str:N", sort=orden_periodos, title=gran),
    y=alt.Y("ticket_promedio:Q", title="Ticket promedio", scale=alt.Scale(zero=False)),
    tooltip=[
        alt.Tooltip("periodo_str:N", title=gran),
        alt.Tooltip("ventas_totales:Q", title="Ventas Totales", format=",.0f"),
        alt.Tooltip("tickets_totales:Q", title="Tickets Totales", format=",.0f"),
        alt.Tooltip("ticket_promedio:Q", title="Ticket promedio", format=",.0f"),
        alt.Tooltip("var_abs:Q", title="Δ abs", format=",.0f"),
        alt.Tooltip("var_pct:Q", title="Δ %", format=".1%"),
        alt.Tooltip("ventas_delivery:Q", title="Ventas Delivery", format=",.0f"),
        alt.Tooltip("tickets_delivery:Q", title="Tickets Delivery", format=",.0f"),
        alt.Tooltip("ticket_promedio_delivery:Q", title="Ticket Prom. Delivery", format=",.0f"),
    ],
)

layers = [line]

if mostrar_rolling and "tp_roll" in serie.columns:
    roll = base.mark_line(strokeDash=[6, 4]).encode(
        x=alt.X("periodo_str:N", sort=orden_periodos),
        y=alt.Y("tp_roll:Q", title=""),
        tooltip=[
            alt.Tooltip("periodo_str:N", title=gran),
            alt.Tooltip("tp_roll:Q", title=f"Rolling {rolling_n}", format=",.0f"),
        ],
    )
    layers.append(roll)

st.altair_chart(alt.layer(*layers).properties(height=330), use_container_width=True)

# =========================================================
# Variación % vs anterior (TOTAL)
# =========================================================
st.markdown("### Incremento / decremento vs periodo anterior (Ticket total)")

var_df = serie.dropna(subset=["var_pct"]).copy()
if var_df.empty:
    st.info("Aún no hay suficientes periodos para calcular variación vs anterior.")
else:
    var_df["var_dir"] = np.where(var_df["var_pct"] >= 0, "Incremento", "Decremento")

    ch_var = (
        alt.Chart(var_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("periodo_str:N", sort=orden_periodos, title=gran),
            y=alt.Y("var_pct:Q", title="Variación (%)", axis=alt.Axis(format="%")),
            color=alt.Color("var_dir:N", legend=alt.Legend(title="Dirección")),
            tooltip=[
                alt.Tooltip("periodo_str:N", title=gran),
                alt.Tooltip("var_pct:Q", title="Variación", format=".1%"),
                alt.Tooltip("var_abs:Q", title="Δ abs", format=",.0f"),
                alt.Tooltip("ticket_promedio:Q", title="Ticket prom.", format=",.0f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(ch_var, use_container_width=True)

st.markdown("---")

# =========================================================
# Insights
# =========================================================
st.markdown("### Insights")
ins1, ins2, ins3 = st.columns(3)

with ins1:
    st.markdown("#### Mejor periodo (ticket total)")
    if best_row is not None:
        st.write(f"**{best_row['periodo_str']}**")
        st.write(f"Ticket prom.: **{fmt_money(best_row['ticket_promedio'])}**")
        st.write(f"Ventas Totales: {fmt_money(best_row['ventas_totales'])} · Tickets: {int(best_row['tickets_totales']):,}")
    else:
        st.write("—")

with ins2:
    st.markdown("#### Peor periodo (ticket total)")
    if worst_row is not None:
        st.write(f"**{worst_row['periodo_str']}**")
        st.write(f"Ticket prom.: **{fmt_money(worst_row['ticket_promedio'])}**")
        st.write(f"Ventas Totales: {fmt_money(worst_row['ventas_totales'])} · Tickets: {int(worst_row['tickets_totales']):,}")
    else:
        st.write("—")

with ins3:
    st.markdown("#### Tendencia reciente (ticket total)")
    recent_n = min(8, len(last_valid))
    if recent_n >= 3:
        sub = last_valid.tail(recent_n).copy()
        y = sub["ticket_promedio"].values.astype(float)
        x = np.arange(len(y), dtype=float)
        slope = np.polyfit(x, y, 1)[0]  # $ por periodo
        st.write(f"Últimos **{recent_n}** periodos:")
        st.write(f"Pendiente aprox.: **{slope:,.1f}** $/periodo")
        if slope > 0:
            st.write("Lectura: tendencia **al alza**.")
        elif slope < 0:
            st.write("Lectura: tendencia **a la baja**.")
        else:
            st.write("Lectura: tendencia **plana**.")
    else:
        st.write("—")

# =========================================================
# Tabla detalle
# =========================================================
st.markdown("### Detalle por periodo")

out = serie[[
    "periodo_str",
    "ventas_totales",
    "tickets_totales",
    "ticket_promedio",
    "ventas_delivery",
    "tickets_delivery",
    "ticket_promedio_delivery",
    "var_abs",
    "var_pct",
]].copy()

out = out.rename(columns={
    "periodo_str": "Periodo",
    "ventas_totales": "Ventas Totales",
    "tickets_totales": "Tickets Totales",
    "ticket_promedio": "Ticket Promedio",
    "ventas_delivery": "Ventas Delivery",
    "tickets_delivery": "Tickets Delivery",
    "ticket_promedio_delivery": "Ticket Promedio Delivery",
    "var_abs": "Δ abs vs ant",
    "var_pct": "Δ % vs ant",
})

st.dataframe(
    out.style.format({
        "Ventas Totales": "${:,.0f}",
        "Ventas Delivery": "${:,.0f}",
        "Ticket Promedio": "${:,.0f}",
        "Ticket Promedio Delivery": "${:,.0f}",
        "Δ abs vs ant": "${:,.0f}",
        "Δ % vs ant": "{:.1%}",
        "Tickets Totales": "{:,.0f}",
        "Tickets Delivery": "{:,.0f}",
    }),
    use_container_width=True
)
