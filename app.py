# app.py — Líneas con semana ISO + Heatmap + Vistas 1/2 + Top Productos & Complementos
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Ops – Ventas", page_icon="📊", layout="wide")

# Fuente de datos (Google Sheets publicado a CSV)
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQZBL6mvIC1OUC-p0MREMW_7UvMKb8It4Y_ldFOi3FbqP4cwZBLrDXwpA_hjBzkeZz3tsOBqd9BlamY/pub?output=csv"

@st.cache_data(ttl=600)
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = [c.strip() for c in df.columns]
    return df

def fmt_money(x): return "—" if pd.isna(x) else f"${x:,.0f}"
def fmt_pct(x):   return "—" if (x is None or pd.isna(x)) else f"{x*100:,.1f}%"

def arrow_gap(p):
    if p is None or pd.isna(p): return "●"   # gris
    if p > 0: return "▲"
    if p < 0: return "▼"
    return "●"

def color_dot_for_target(p):
    if p is None or pd.isna(p): return "⚪"
    if p >= 1.0: return "🟢"
    if p >= 0.80: return "🟡"
    return "🔴"

# ---- Carga
df = load_data()

# ---- Columnas (nombres que nos diste)
COL_CC, COL_FECHA, COL_SUBTOT, COL_TIPO, COL_FOLIO = "Restaurante", "Fecha", "Subtotal", "Tipo", "Folio"
COL_DETALLE = "Detalle Items"  # para Top Productos & Complementos

# ---- Tipificación
df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
df[COL_SUBTOT] = df[COL_SUBTOT].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)

# =======================
#   HELPERS GRANULARIDAD
# =======================
def add_periodo(d: pd.Series, gran: str) -> pd.Series:
    """Devuelve la marca temporal que se usará como x según granularidad."""
    if gran == "Mes":
        return d.dt.to_period("M").dt.to_timestamp()
    if gran == "Semana":
        # Usamos el inicio de semana (lunes) como timestamp del punto
        return d.dt.to_period("W").dt.start_time
    return d.dt.to_period("D").dt.to_timestamp()

def axis_for(gran: str) -> alt.Axis:
    """Formatea el eje X."""
    if gran == "Mes":
        return alt.Axis(title="Mes", format="%b %Y", labelAngle=0)
    if gran == "Semana":
        # %V = número de semana ISO; mostramos W36 2025
        return alt.Axis(title="Semana ISO", format="W%V %Y", labelAngle=0)
    return alt.Axis(title="Día", format="%d %b", labelAngle=0)

def es_delivery(val):
    try:    return "delivery" in str(val).lower()
    except: return False

# ============================================================
#  Helper para ventanas por granularidad (Día / Semana ISO / Mes)
# ============================================================
def period_windows(end_ts: pd.Timestamp, gran: str):
    end_ts = pd.to_datetime(end_ts)

    if gran == "Mes":
        p = end_ts.to_period("M")
        cur_ini, cur_fin = p.start_time, p.end_time
        prev_p = (p - 1)
        prev_ini, prev_fin = prev_p.start_time, prev_p.end_time
        title = f"{p.strftime('%B %Y').title()} (1 al {cur_fin.day})"

    elif gran == "Semana":
        # ventana = semana ISO que contiene end_ts (lunes a domingo)
        d = end_ts.normalize()
        cur_ini = d - pd.Timedelta(days=d.weekday())  # lunes
        cur_fin = cur_ini + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59, milliseconds=999)
        prev_ini, prev_fin = cur_ini - pd.Timedelta(days=7), cur_fin - pd.Timedelta(days=7)
        iso = d.isocalendar()
        title = f"Semana ISO W{int(iso.week)} {int(iso.year)}"

    else:  # "Día"
        d = end_ts.normalize()
        cur_ini, cur_fin = d, d + pd.Timedelta(hours=23, minutes=59, seconds=59, milliseconds=999)
        prev_ini, prev_fin = d - pd.Timedelta(days=1), cur_fin - pd.Timedelta(days=1)
        title = f"Día {d.date()}"

    return cur_ini, cur_fin, prev_ini, prev_fin, title

# =======================
#        SIDEBAR
# =======================
st.sidebar.header("Filtros")

if df[COL_FECHA].notna().any():
    min_f, max_f = df[COL_FECHA].min(), df[COL_FECHA].max()
    rango = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_f.date() if pd.notna(min_f) else datetime(2025,1,1).date(),
               max_f.date() if pd.notna(max_f) else datetime.today().date()),
    )
    if isinstance(rango, tuple):
        f_ini, f_fin = [pd.to_datetime(x) for x in rango]
    else:
        f_ini, f_fin = min_f, max_f
else:
    f_ini = f_fin = None

rest_all = sorted(df[COL_CC].dropna().astype(str).unique())
rest_sel = st.sidebar.multiselect("Restaurante(s)", rest_all, default=rest_all)

granularidad = st.sidebar.radio("Granularidad", ["Día","Semana","Mes"], index=2, horizontal=True)

if f_fin is None:
    st.stop()

# =======================
#  DATA FILTRADA GLOBAL
# =======================
f = df[(df[COL_FECHA] >= f_ini) & (df[COL_FECHA] <= f_fin)].copy()
if rest_sel:
    f = f[f[COL_CC].astype(str).isin(rest_sel)]

# =======================
#        RESUMEN
# =======================
ventas_total = f[COL_SUBTOT].sum()
n_tickets = f[COL_FOLIO].nunique() if COL_FOLIO in f.columns else f.shape[0]
aov = (ventas_total / n_tickets) if (ventas_total and n_tickets) else None

f["_mes"] = f[COL_FECHA].dt.to_period("M").dt.to_timestamp()
var_mes_anterior = None
meses = sorted(f["_mes"].dropna().unique())
if len(meses) >= 2:
    m_act, m_prev = meses[-1], meses[-2]
    v_act = f.loc[f["_mes"]==m_act, COL_SUBTOT].sum()
    v_prev= f.loc[f["_mes"]==m_prev, COL_SUBTOT].sum()
    if v_prev: var_mes_anterior = (v_act / v_prev) - 1

st.markdown("## Resumen")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Ventas Netas", fmt_money(ventas_total))
c2.metric("Tickets (folios únicos)", f"{n_tickets:,.0f}" if n_tickets else "—")
c3.metric("Ticket Promedio", fmt_money(aov) if aov else "—")
c4.metric("% vs Mes Anterior", fmt_pct(var_mes_anterior) if var_mes_anterior is not None else "—")

# =======================
#     PESTAÑAS GRÁFICAS
# =======================
tab_ventas, tab_trans, tab_heat, tab_top = st.tabs(
    ["📈 Ventas", "🧾 Transacciones", "🗓️ Heatmap D/H", "🥇 Top Prod & Complem"]
)

# --- VENTAS (línea)
with tab_ventas:
    g = f.copy()
    g["periodo"] = add_periodo(g[COL_FECHA], granularidad)
    iso = g[COL_FECHA].dt.isocalendar()
    g["semana_iso"] = iso.week.astype(int)
    g["anio_iso"]   = iso.year.astype(int)

    grp_cols = ["periodo", COL_CC] if len(rest_sel) > 1 else ["periodo"]
    v = g.groupby(grp_cols, as_index=False)[COL_SUBTOT].sum()

    if v.empty:
        st.info("Sin datos en el rango.")
    else:
        base = alt.Chart(v).mark_line(point=True).encode(
            x=alt.X("periodo:T", axis=axis_for(granularidad)),
            y=alt.Y(f"{COL_SUBTOT}:Q", title="Ventas"),
            tooltip=[
                "periodo:T",
                alt.Tooltip(f"{COL_SUBTOT}:Q", format=",", title="Ventas"),
                alt.Tooltip("semana_iso:Q", title="Semana ISO"),
                alt.Tooltip("anio_iso:Q",   title="Año ISO"),
            ],
        )
        if len(rest_sel) > 1:
            ch = base.encode(color=alt.Color(f"{COL_CC}:N", title="Restaurante"),
                             tooltip=["periodo:T", f"{COL_CC}:N",
                                      alt.Tooltip(f"{COL_SUBTOT}:Q", format=",", title="Ventas"),
                                      alt.Tooltip("semana_iso:Q", title="Semana ISO"),
                                      alt.Tooltip("anio_iso:Q",   title="Año ISO")])
        else:
            ch = base
        st.altair_chart(ch.properties(height=320), use_container_width=True)

# --- TRANSACCIONES (línea) Tienda vs Delivery
with tab_trans:
    g = f.copy()
    g["periodo"] = add_periodo(g[COL_FECHA], granularidad)
    iso = g[COL_FECHA].dt.isocalendar()
    g["semana_iso"] = iso.week.astype(int)
    g["anio_iso"]   = iso.year.astype(int)
    g["_is_delivery"] = g[COL_TIPO].apply(es_delivery)
    g["canal"] = np.where(g["_is_delivery"], "Delivery", "Tienda")

    if COL_FOLIO in g.columns:
        t = g.groupby(["periodo","canal"], as_index=False)[COL_FOLIO].nunique().rename(columns={COL_FOLIO:"Transacciones"})
    else:
        t = g.groupby(["periodo","canal"], as_index=False).size().rename(columns={"size":"Transacciones"})

    if t.empty:
        st.info("Sin datos en el rango.")
    else:
        ch = alt.Chart(t).mark_line(point=True).encode(
            x=alt.X("periodo:T", axis=axis_for(granularidad)),
            y=alt.Y("Transacciones:Q"),
            color=alt.Color("canal:N", title="Canal"),
            tooltip=["periodo:T","canal:N",
                     alt.Tooltip("Transacciones:Q", format=","),
                     alt.Tooltip("semana_iso:Q", title="Semana ISO"),
                     alt.Tooltip("anio_iso:Q",   title="Año ISO")]
        )
        st.altair_chart(ch.properties(height=320), use_container_width=True)

# --- HEATMAP Día/Hora (ventas)
with tab_heat:
    if f.empty:
        st.info("Sin datos en el rango.")
    else:
        g = f.copy()
        g["dow"]  = g[COL_FECHA].dt.day_name()
        g["hora"] = g[COL_FECHA].dt.hour
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        g["dow"] = pd.Categorical(g["dow"], categories=order, ordered=True)
        hm = g.groupby(["dow","hora"], as_index=False)[COL_SUBTOT].sum()

        ch = alt.Chart(hm).mark_rect().encode(
            x=alt.X("hora:O", title="Hora"),
            y=alt.Y("dow:O",  title="Día"),
            color=alt.Color(f"{COL_SUBTOT}:Q", title="Ventas"),
            tooltip=["dow:O","hora:O", alt.Tooltip(f"{COL_SUBTOT}:Q", format=",", title="Ventas")]
        )
        st.altair_chart(ch.properties(height=320), use_container_width=True)

# ============================
#   PARSER Detalle Items
# ============================
def _norm_txt(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if s.startswith("+"):
        s = s[1:].strip()
    return s

def _extract_qty(seg: str) -> int:
    m = re.search(r"x\s*(\d+)", seg, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 1

def _drop_price(seg: str) -> str:
    return re.sub(r"\(\s*[^)]*\)", "", seg).strip()

def _clean_product(seg: str) -> str:
    seg = _drop_price(seg)
    seg = re.sub(r"x\s*\d+", "", seg, flags=re.IGNORECASE)  # quita "xN"
    seg = _norm_txt(seg)
    return seg

def parse_detalle_items(raw: str):
    """
    Recibe todo el 'Detalle Items' de una orden y devuelve:
      - lista de (producto, qty)
      - lista de (complemento, qty)
    Reglas:
      - Ítems separados por '|'
      - Complementos dentro de [ ... ] separados por comas
      - Cantidad (xN) multiplica producto y sus complementos
      - Precios dentro de (...) se ignoran para conteo
    """
    productos = []
    compls = []

    if not raw or not isinstance(raw, str):
        return productos, compls

    partes = [p.strip() for p in re.split(r"\s*\|\s*", raw) if p.strip()]
    for parte in partes:
        qty = _extract_qty(parte)

        comps = []
        m = re.search(r"\[(.*?)\]", parte)
        if m:
            dentro = m.group(1)
            comps = [_norm_txt(c) for c in dentro.split(",") if _norm_txt(c)]

        prod_seg = re.sub(r"\[.*?\]", "", parte).strip()
        prod_name = _clean_product(prod_seg)
        if prod_name:
            productos.append((prod_name, qty))

        for c in comps:
            compls.append((c, qty))

    return productos, compls

# --- TOP Productos & Complementos (por unidades)
with tab_top:
    st.subheader("Top 10 · Productos y Complementos (unidades)")

    limitar_a_periodo = st.checkbox(
        "Limitar al periodo actual según granularidad (Día/Semana ISO/Mes)",
        value=True
    )

    # Base ya filtrada por rango y restaurantes
    f_top = f.copy()

    if limitar_a_periodo:
        ini_p, fin_p, _, _, title_periodo = period_windows(f_fin, granularidad)
        st.caption(f"Periodo actual: **{title_periodo}**  |  {ini_p.strftime('%Y-%m-%d')} → {fin_p.strftime('%Y-%m-%d')}")
        f_top = f_top[(f_top[COL_FECHA] >= ini_p) & (f_top[COL_FECHA] <= fin_p)]
    else:
        st.caption("Periodo: rango completo de fechas seleccionado en la barra lateral.")

    if f_top.empty or COL_DETALLE not in f_top.columns:
        st.info("No hay datos (con los filtros/periodo) o falta la columna 'Detalle Items'.")
    else:
        prod_counts, comp_counts = {}, {}

        for raw in f_top[COL_DETALLE].dropna().astype(str):
            prods, comps = parse_detalle_items(raw)
            for name, q in prods:
                if not name:
                    continue
                prod_counts[name] = prod_counts.get(name, 0) + int(q)
            for name, q in comps:
                if not name:
                    continue
                comp_counts[name] = comp_counts.get(name, 0) + int(q)

        df_prod = (pd.DataFrame(list(prod_counts.items()), columns=["Producto","Unidades"])
                     .sort_values("Unidades", ascending=False)
                     .head(10))
        df_comp = (pd.DataFrame(list(comp_counts.items()), columns=["Complemento","Unidades"])
                     .sort_values("Unidades", ascending=False)
                     .head(10))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 Productos**")
            if not df_prod.empty:
                ch_p = alt.Chart(df_prod).mark_bar().encode(
                    x=alt.X("Unidades:Q", title="Unidades"),
                    y=alt.Y("Producto:N", sort='-x', title=None),
                    tooltip=["Producto:N", alt.Tooltip("Unidades:Q", format=",")]
                ).properties(height=360)
                st.altair_chart(ch_p, use_container_width=True)
                st.dataframe(df_prod, use_container_width=True)
            else:
                st.info("Sin productos para el periodo/selección.")

        with c2:
            st.markdown("**Top 10 Complementos**")
            if not df_comp.empty:
                ch_c = alt.Chart(df_comp).mark_bar().encode(
                    x=alt.X("Unidades:Q", title="Unidades"),
                    y=alt.Y("Complemento:N", sort='-x', title=None),
                    tooltip=["Complemento:N", alt.Tooltip("Unidades:Q", format=",")]
                ).properties(height=360)
                st.altair_chart(ch_c, use_container_width=True)
                st.dataframe(df_comp, use_container_width=True)
            else:
                st.info("Sin complementos para el periodo/selección.")

# ============================================================
#  VISTAS 1/2 (comparativo periodo actual vs anterior)
#  Usan la granularidad elegida (día/semana/mes) anclando al fin del rango.
# ============================================================
ini_p, fin_p, ini_ant, fin_ant, title_periodo = period_windows(f_fin, granularidad)

base_cur  = df[(df[COL_FECHA] >= ini_p) & (df[COL_FECHA] <= fin_p)].copy()
base_prev = df[(df[COL_FECHA] >= ini_ant) & (df[COL_FECHA] <= fin_ant)].copy()
if rest_sel:
    base_cur  = base_cur [base_cur [COL_CC].astype(str).isin(rest_sel)]
    base_prev = base_prev[base_prev[COL_CC].astype(str).isin(rest_sel)]

st.markdown(f"## Ventas – {title_periodo}")

# ========== VISTA 1: Ventas Netas por Marca ==========
st.subheader("Ventas Netas por Marca")
v_cur  = base_cur.groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Ventas Periodo"})
v_prev = base_prev.groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT: "Ventas Periodo Ant"})
vista1 = v_cur.merge(v_prev, on=COL_CC, how="left")
vista1["Budget"] = vista1["Ventas Periodo"] * 1.05
vista1["% Alcance vs Budget"] = np.where(vista1["Budget"]>0, vista1["Ventas Periodo"]/vista1["Budget"], np.nan)

def calc_gap(vm, va):
    if pd.isna(va) or va==0: return None
    return (vm/va)-1

vista1["% GAP vs Periodo Ant"] = [calc_gap(vm,va) for vm,va in zip(vista1["Ventas Periodo"], vista1["Ventas Periodo Ant"])]
vista1 = vista1.sort_values("Ventas Periodo", ascending=False)

tot_vm = vista1["Ventas Periodo"].sum()
tot_va = vista1["Ventas Periodo Ant"].sum()
tot_budget  = tot_vm * 1.05
tot_alcance = (tot_vm / tot_budget) if tot_budget else np.nan
tot_gap     = calc_gap(tot_vm, tot_va)

tot_row = pd.DataFrame({
    COL_CC: ["Total Tiendas"],
    "Ventas Periodo": [tot_vm], "Ventas Periodo Ant": [tot_va],
    "Budget": [tot_budget], "% Alcance vs Budget": [tot_alcance], "% GAP vs Periodo Ant": [tot_gap],
})
vista1 = pd.concat([vista1, tot_row], ignore_index=True)

out1 = pd.DataFrame({
    "CC": vista1[COL_CC],
    "Ventas Netas (Periodo)": vista1["Ventas Periodo"].apply(fmt_money),
    "Budget (Periodo)": vista1["Budget"].apply(fmt_money),
    "% Alcance Vs Budget": [f"{fmt_pct(p)} {color_dot_for_target(p)}" for p in vista1["% Alcance vs Budget"]],
    "% GAP vs Periodo Anterior": [f"{fmt_pct(p)} {arrow_gap(p)}" for p in vista1["% GAP vs Periodo Ant"]],
})
st.dataframe(out1, use_container_width=True)

# ========== VISTA 2: Ventas Delivery por Marca ==========
st.subheader("Ventas Delivery por Marca")
base_cur["__is_delivery"]  = base_cur[COL_TIPO].apply(es_delivery)
base_prev["__is_delivery"] = base_prev[COL_TIPO].apply(es_delivery)

tienda_cur   = base_cur.loc[~base_cur["__is_delivery"]].groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT:"Ventas Tienda"})
delivery_cur = base_cur.loc[ base_cur["__is_delivery"]].groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT:"Ventas Delivery"})
tot_cur      = base_cur.groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT:"Ventas Totales"})
delivery_prev= base_prev.loc[base_prev["__is_delivery"]].groupby(COL_CC, as_index=False)[COL_SUBTOT].sum().rename(columns={COL_SUBTOT:"Delivery Periodo Ant"})

vista2 = (tot_cur.merge(tienda_cur, on=COL_CC, how="left")
                .merge(delivery_cur, on=COL_CC, how="left")
                .merge(delivery_prev,on=COL_CC, how="left"))

vista2[["Ventas Tienda","Ventas Delivery","Ventas Totales","Delivery Periodo Ant"]] = \
    vista2[["Ventas Tienda","Ventas Delivery","Ventas Totales","Delivery Periodo Ant"]].fillna(0)

vista2["% Tienda"]   = np.where(vista2["Ventas Totales"]>0, vista2["Ventas Tienda"]/vista2["Ventas Totales"], np.nan)
vista2["% Delivery"] = np.where(vista2["Ventas Totales"]>0, vista2["Ventas Delivery"]/vista2["Ventas Totales"], np.nan)

def pct_vs_prev(cur, ant):
    if ant==0: return None
    return (cur/ant)-1

vista2["% vs Periodo Ant"] = [pct_vs_prev(c,a) for c,a in zip(vista2["Ventas Delivery"], vista2["Delivery Periodo Ant"])]
vista2 = vista2.sort_values("Ventas Totales", ascending=False)

tot2_vm_tienda  = vista2["Ventas Tienda"].sum()
tot2_vm_deliv   = vista2["Ventas Delivery"].sum()
tot2_vm_total   = vista2["Ventas Totales"].sum()
tot2_vm_antdel  = vista2["Delivery Periodo Ant"].sum()
tot2_pct_tienda = (tot2_vm_tienda/tot2_vm_total) if tot2_vm_total else None
tot2_pct_deliv  = (tot2_vm_deliv /tot2_vm_total) if tot2_vm_total else None
tot2_pct_prev   = ((tot2_vm_deliv/tot2_vm_antdel)-1) if tot2_vm_antdel else None

tot2 = pd.DataFrame([{
    COL_CC:"Total Tiendas",
    "Ventas Tienda":tot2_vm_tienda, "Ventas Delivery":tot2_vm_deliv,
    "Ventas Totales":tot2_vm_total, "Delivery Periodo Ant":tot2_vm_antdel,
    "% Tienda":tot2_pct_tienda, "% Delivery":tot2_pct_deliv, "% vs Periodo Ant":tot2_pct_prev
}])
vista2 = pd.concat([vista2, tot2], ignore_index=True)

out2 = pd.DataFrame({
    "CC": vista2[COL_CC],
    "Ventas Netas Tienda": vista2["Ventas Tienda"].apply(fmt_money),
    "% Tienda": vista2["% Tienda"].apply(fmt_pct),
    "Ventas Delivery": vista2["Ventas Delivery"].apply(fmt_money),
    "% Delivery": vista2["% Delivery"].apply(fmt_pct),
    "Delivery Periodo Anterior": vista2["Delivery Periodo Ant"].apply(fmt_money),
    "% vs Periodo Anterior": [f"{fmt_pct(p)} {arrow_gap(p)}" for p in vista2["% vs Periodo Ant"]],
})
st.dataframe(out2, use_container_width=True)

st.caption("Las pestañas agrupan y formatean el eje X según la granularidad: Mes (%b %Y), Semana ISO (W%V %Y) y Día (%d %b). "
           "Vistas 1/2 comparan periodo actual vs anterior; Budget = 1.05× ventas. "
           "La pestaña de Top parsea 'Detalle Items' para productos y complementos y respeta rango/granularidad (checkbox).")
