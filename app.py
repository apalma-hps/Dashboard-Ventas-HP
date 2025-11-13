# app.py — versión endurecida para Streamlit Cloud
import re
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Ventas BYF W&C", page_icon="📊", layout="wide")

# ------------ CONFIG ------------
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQZBL6mvIC1OUC-p0MREMW_7UvMKb8It4Y_ldFOi3FbqP4cwZBLrDXwpA_hjBzkeZz3tsOBqd9BlamY/pub?output=csv"

COL_CC     = "Restaurante"
COL_FECHA  = "Fecha"
COL_SUBTOT = "Subtotal"
COL_TIPO   = "Tipo"
COL_FOLIO  = "Folio"
COL_DET    = "Detalle Items"

# ------------ UTILS ------------
@st.cache_data(ttl=600)
def load_raw_csv(url: str) -> pd.DataFrame:
    # Lee CSV con encoding defensivo
    try:
        df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv(url, encoding="latin-1")
    # strip columnas
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_fecha_series(s: pd.Series) -> pd.Series:
    # Intento 1: dayfirst (común en MX)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    # Si demasiado NaN, reintenta monthfirst
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False, utc=False)
    return dt

_num_re = re.compile(r"[^\d\-\.,()]")  # quita todo excepto dígitos, punto, coma, guión, paréntesis

def parse_money_series(s: pd.Series) -> pd.Series:
    # elimina símbolos y espacios
    x = s.astype(str).str.replace(_num_re, "", regex=True).str.strip()
    # maneja paréntesis como negativos
    neg = x.str.contains(r"\(") & x.str.contains(r"\)")
    x = x.str.replace("[()]", "", regex=True)
    # si hay ambos punto y coma, asume coma como sep de miles y punto decimal
    both = x.str.contains(",") & x.str.contains(r"\.")
    x = np.where(both, x.str.replace(",", "", regex=False), x)
    # si sólo hay comas, asume decimal con coma → cambia a punto
    only_comma = x.str.contains(",") & ~x.str.contains(r"\.")
    x = np.where(only_comma, x.str.replace(",", ".", regex=False), x)
    x = pd.to_numeric(x, errors="coerce")
    x = np.where(neg, -x, x)
    return pd.to_numeric(x, errors="coerce")

def fmt_money(v):
    return "—" if pd.isna(v) else f"${v:,.0f}"

def fmt_pct(p):
    return "—" if (p is None or pd.isna(p)) else f"{p*100:,.1f}%"

def is_delivery(val) -> bool:
    try:
        s = str(val).lower()
        return "delivery" in s
    except Exception:
        return False

# ------------ DATA ------------
st.sidebar.markdown("### Actualización")
if st.sidebar.button("Refrescar datos"):
    st.cache_data.clear()
    st.rerun()

df = load_raw_csv(DATA_URL)

# Verificamos que existan las columnas esperadas
expected = [COL_CC, COL_FECHA, COL_SUBTOT, COL_TIPO]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el dataset: {missing}")
    st.stop()

# Parseo robusto
df[COL_FECHA] = parse_fecha_series(df[COL_FECHA])
df[COL_SUBTOT] = parse_money_series(df[COL_SUBTOT])

# Limpia filas sin fecha o subtotal
df = df[df[COL_FECHA].notna() & df[COL_SUBTOT].notna()].copy()

# ------------ FILTROS ------------
st.sidebar.header("Filtros")

min_f = pd.to_datetime(df[COL_FECHA]).min()
max_f = pd.to_datetime(df[COL_FECHA]).max()

rango = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_f.date(), max_f.date()),
)

if isinstance(rango, tuple):
    f_ini, f_fin = [pd.to_datetime(x) for x in rango]
else:
    f_ini, f_fin = min_f, max_f

# Filtro por marca/restaurante si existe
marcas = sorted(df[COL_CC].dropna().astype(str).unique())
marca_sel = st.sidebar.multiselect("Restaurante", marcas, default=marcas)

granularidad = st.sidebar.radio("Granularidad", ["Día", "Semana", "Mes"], index=2, horizontal=True)

# Aplicar filtros
f = df[
    (df[COL_FECHA].between(f_ini, f_fin))
    & (df[COL_CC].astype(str).isin(marca_sel))
].copy()

# ------------ KPI RESUMEN ------------
st.subheader("Resumen")

ventas_total = f[COL_SUBTOT].sum()
n_tickets = f[COL_FOLIO].nunique() if COL_FOLIO in f.columns else f.shape[0]
aov = (ventas_total / n_tickets) if n_tickets else np.nan

# Variación vs periodo anterior (misma duración)
span_days = (f_fin - f_ini).days + 1
prev_ini = f_ini - pd.Timedelta(days=span_days)
prev_fin = f_ini - pd.Timedelta(days=1)
f_prev = df[
    (df[COL_FECHA].between(prev_ini, prev_fin))
    & (df[COL_CC].astype(str).isin(marca_sel))
].copy()
ventas_prev = f_prev[COL_SUBTOT].sum()
var_vs_prev = ((ventas_total / ventas_prev) - 1) if ventas_prev else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ventas Netas", fmt_money(ventas_total))
c2.metric("Tickets", f"{n_tickets:,.0f}")
c3.metric("Ticket Promedio", fmt_money(aov))
c4.metric("Var. vs periodo anterior", fmt_pct(var_vs_prev))

# ------------ TENDENCIAS ------------
st.subheader("Tendencias")

g = f.copy()
if granularidad == "Día":
    g["periodo"] = g[COL_FECHA].dt.to_period("D").dt.to_timestamp()
elif granularidad == "Semana":
    iso = g[COL_FECHA].dt.isocalendar()
    # fecha del lunes de esa semana ISO
    g["periodo"] = (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2) + "-1")
    g["periodo"] = pd.to_datetime(g["periodo"])
else:
    g["periodo"] = g[COL_FECHA].dt.to_period("M").dt.to_timestamp()

ser = g.groupby("periodo", as_index=False)[COL_SUBTOT].sum().sort_values("periodo")

import altair as alt
line_total = alt.Chart(ser).mark_line(point=True).encode(
    x=alt.X("periodo:T", title="Periodo"),
    y=alt.Y(f"{COL_SUBTOT}:Q", title="Ventas"),
    tooltip=[alt.Tooltip("periodo:T", title="Periodo"),
             alt.Tooltip(f"{COL_SUBTOT}:Q", title="Ventas", format=",")]
).properties(height=320)

st.altair_chart(line_total, use_container_width=True)

# ------------ TIENDA vs DELIVERY (por periodo) ------------
st.subheader("Tienda vs Delivery")

g["is_delivery"] = g[COL_TIPO].map(is_delivery)
ser2 = g.groupby(["periodo", "is_delivery"], as_index=False)[COL_SUBTOT].sum()
ser2["canal"] = np.where(ser2["is_delivery"], "Delivery", "Tienda")

line_split = alt.Chart(ser2).mark_line(point=True).encode(
    x=alt.X("periodo:T", title="Periodo"),
    y=alt.Y(f"{COL_SUBTOT}:Q", title="Ventas"),
    color=alt.Color("canal:N", title="Canal"),
    tooltip=[
        alt.Tooltip("periodo:T", title="Periodo"),
        alt.Tooltip("canal:N", title="Canal"),
        alt.Tooltip(f"{COL_SUBTOT}:Q", title="Ventas", format=",")
    ]
).properties(height=320)

st.altair_chart(line_split, use_container_width=True)

# ------------ VISTAS MENSUALES (budget +5% y delivery detalle) ------------
st.subheader("Vistas Mensuales (para el último mes del rango)")

# Tomamos el mes del f_fin
mes_sel = pd.Period(f_fin, freq="M")
ini_m = mes_sel.to_timestamp(how="start")
fin_m = mes_sel.to_timestamp(how="end")
mes_ant = mes_sel - 1
ini_m_ant = mes_ant.to_timestamp(how="start")
fin_m_ant = mes_ant.to_timestamp(how="end")

sub_m = f[(f[COL_FECHA].between(ini_m, fin_m))].copy()
sub_m_ant = df[
    (df[COL_FECHA].between(ini_m_ant, fin_m_ant))
    & (df[COL_CC].astype(str).isin(marca_sel))
].copy()

# Vista 1: Ventas por CC + Budget
v_mes = sub_m.groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Ventas Mes"})
v_ant = sub_m_ant.groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Ventas Mes Ant"})
vista1 = v_mes.merge(v_ant, on=COL_CC, how="left")
vista1["Budget"] = vista1["Ventas Mes"] * 1.05

def _gap(cur, ant):
    if pd.isna(ant) or ant == 0:
        return np.nan
    return (cur / ant) - 1

vista1["% Alcance vs Budget"] = np.where(vista1["Budget"] > 0, vista1["Ventas Mes"] / vista1["Budget"], np.nan)
vista1["% GAP vs Mes Ant"] = [_gap(c, a) for c, a in zip(vista1["Ventas Mes"], vista1["Ventas Mes Ant"])]

vista1 = vista1.sort_values("Ventas Mes", ascending=False)
tot_row = pd.DataFrame({
    COL_CC: ["Total Tiendas"],
    "Ventas Mes": [vista1["Ventas Mes"].sum()],
    "Ventas Mes Ant": [vista1["Ventas Mes Ant"].sum()],
    "Budget": [vista1["Budget"].sum()],
    "% Alcance vs Budget": [vista1["Ventas Mes"].sum() / vista1["Budget"].sum() if vista1["Budget"].sum() else np.nan],
    "% GAP vs Mes Ant": [_gap(vista1["Ventas Mes"].sum(), vista1["Ventas Mes Ant"].sum())],
})
vista1 = pd.concat([vista1, tot_row], ignore_index=True)

out1 = pd.DataFrame({
    "CC": vista1[COL_CC],
    "Ventas Netas (Mes)": vista1["Ventas Mes"].map(fmt_money),
    "Budget (Mes)": vista1["Budget"].map(fmt_money),
    "% Alcance Vs Budget": vista1["% Alcance vs Budget"].map(fmt_pct),
    "% GAP vs Mes Anterior": vista1["% GAP vs Mes Ant"].map(fmt_pct),
})
st.dataframe(out1, use_container_width=True)

# Vista 2: Delivery por CC
sub_m["__is_delivery"] = sub_m[COL_TIPO].map(is_delivery)
sub_m_ant["__is_delivery"] = sub_m_ant[COL_TIPO].map(is_delivery)

tienda_mes = sub_m.loc[~sub_m["__is_delivery"]].groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Ventas Tienda"})
delivery_mes = sub_m.loc[sub_m["__is_delivery"]].groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Ventas Delivery"})
tot_mes = sub_m.groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Ventas Totales"})
delivery_ant = sub_m_ant.loc[sub_m_ant["__is_delivery"]].groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Delivery Mes Ant"})

vista2 = (tot_mes.merge(tienda_mes, on=COL_CC, how="left")
                .merge(delivery_mes, on=COL_CC, how="left")
                .merge(delivery_ant, on=COL_CC, how="left"))

for c in ["Ventas Tienda", "Ventas Delivery", "Ventas Totales", "Delivery Mes Ant"]:
    if c not in vista2.columns:
        vista2[c] = 0.0
vista2[["Ventas Tienda", "Ventas Delivery", "Ventas Totales", "Delivery Mes Ant"]] = vista2[["Ventas Tienda", "Ventas Delivery", "Ventas Totales", "Delivery Mes Ant"]].fillna(0)

vista2["% Tienda"] = np.where(vista2["Ventas Totales"] > 0, vista2["Ventas Tienda"] / vista2["Ventas Totales"], np.nan)
vista2["% Delivery"] = np.where(vista2["Ventas Totales"] > 0, vista2["Ventas Delivery"] / vista2["Ventas Totales"], np.nan)

def _var(cur, ant):
    if ant == 0:
        return np.nan
    return (cur / ant) - 1

vista2["% Mes Anterior"] = [_var(c, a) for c, a in zip(vista2["Ventas Delivery"], vista2["Delivery Mes Ant"])]
vista2 = vista2.sort_values("Ventas Totales", ascending=False)

tot2 = pd.DataFrame([{
    COL_CC: "Total Tiendas",
    "Ventas Tienda": vista2["Ventas Tienda"].sum(),
    "Ventas Delivery": vista2["Ventas Delivery"].sum(),
    "Ventas Totales": vista2["Ventas Totales"].sum(),
    "Delivery Mes Ant": vista2["Delivery Mes Ant"].sum(),
    "% Tienda": (vista2["Ventas Tienda"].sum() / vista2["Ventas Totales"].sum()) if vista2["Ventas Totales"].sum() else np.nan,
    "% Delivery": (vista2["Ventas Delivery"].sum() / vista2["Ventas Totales"].sum()) if vista2["Ventas Totales"].sum() else np.nan,
    "% Mes Anterior": _var(vista2["Ventas Delivery"].sum(), vista2["Delivery Mes Ant"].sum())
}])
vista2 = pd.concat([vista2, tot2], ignore_index=True)

out2 = pd.DataFrame({
    "CC": vista2[COL_CC],
    "Ventas Netas Tienda": vista2["Ventas Tienda"].map(fmt_money),
    "% Tienda": vista2["% Tienda"].map(fmt_pct),
    "Ventas Delivery": vista2["Ventas Delivery"].map(fmt_money),
    "% Delivery": vista2["% Delivery"].map(fmt_pct),
    "Delivery Mes Anterior": vista2["Delivery Mes Ant"].map(fmt_money),
    "% Mes Anterior": vista2["% Mes Anterior"].map(fmt_pct),
})
st.dataframe(out2, use_container_width=True)
