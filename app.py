# app.py — Dashboard Ventas BYF W&C (Streamlit + Google Sheets)
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
import re

st.set_page_config(page_title="Ops – Ventas", page_icon="📊", layout="wide")

# URL pública de Google Sheets (CSV)
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQZBL6mvIC1OUC-p0MREMW_7UvMKb8It4Y_ldFOi3FbqP4cwZBLrDXwpA_hjBzkeZz3tsOBqd9BlamY/pub?output=csv"


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



# ---------- Carga de datos (con botón de refresco) ----------
st.sidebar.header("Datos")

if st.sidebar.button("🔄 Actualizar datos de Google Sheets"):
    # Limpia la caché de load_data y vuelve a correr el script
    load_data.clear()
    st.experimental_rerun()

# Ahora sí cargamos los datos (después del botón)
df = load_data()

# Columnas base (usa EXACTAMENTE estos nombres)
COL_CC = "Restaurante"     # CC / Marca
COL_FECHA = "Fecha"        # fecha de la venta
COL_SUBTOT = "Subtotal"    # ventas netas
COL_TIPO = "Tipo"          # detectar delivery
COL_FOLIO = "Folio"        # tickets
COL_DETALLE = "Detalle Items"  # para top productos / complementos
COL_VENTAS = COL_SUBTOT

# Tipificación mínima
df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)

df[COL_SUBTOT] = (
    df[COL_SUBTOT]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# ---------- Sidebar: filtros ----------
st.sidebar.header("Filtros")


if df[COL_FECHA].notna().any():
    min_f = df[COL_FECHA].min()
    max_f = df[COL_FECHA].max()
    rango = st.sidebar.date_input(
        "Rango de fechas",
        value=(
            min_f.date() if pd.notna(min_f) else datetime(2025, 1, 1).date(),
            max_f.date() if pd.notna(max_f) else datetime.today().date(),
        ),
    )

    # date_input puede regresar una fecha o una tupla
    if isinstance(rango, (list, tuple)):
        if len(rango) == 2:
            f_ini, f_fin = [pd.to_datetime(x) for x in rango]
        else:
            f_ini = f_fin = pd.to_datetime(rango[0])
    else:
        f_ini = f_fin = pd.to_datetime(rango)
else:
    f_ini = f_fin = None

# Filtro por restaurante / marca
rest_sel = None
if COL_CC in df.columns and df[COL_CC].notna().any():
    rests = sorted(df[COL_CC].dropna().astype(str).unique())
    rest_sel = st.sidebar.multiselect("Restaurante", rests, default=rests)

# Granularidad SOLO para las gráficas de tendencia
granularidad = st.sidebar.radio(
    "Agrupar tendencia por",
    ["Día", "Semana", "Mes"],
    index=0,
    horizontal=True,
)

# ---------- Aplicar filtros base al df filtrado (f) ----------
f = df.copy()
if COL_FECHA and f_ini is not None and f_fin is not None:
    ini_d, fin_d = f_ini.date(), f_fin.date()
    mask = (f[COL_FECHA].dt.date >= ini_d) & (f[COL_FECHA].dt.date <= fin_d)
    f = f[mask]

if rest_sel is not None:
    f = f[f[COL_CC].astype(str).isin(rest_sel)]

# Si no hay datos, salimos limpio
if f.empty:
    st.warning("No hay datos en el rango de fechas / restaurantes seleccionados.")
    st.stop()

# ---------- Definir mes seleccionado y mes anterior (para vistas mensuales) ----------
if f_fin is None:
    st.stop()

mes_sel = pd.Period(f_fin, freq="M")
ini_mes = mes_sel.to_timestamp(how="start")
fin_mes = mes_sel.to_timestamp(how="end")

mes_ant = mes_sel - 1
ini_mes_ant = mes_ant.to_timestamp(how="start")
fin_mes_ant = mes_ant.to_timestamp(how="end")

# Subconjuntos mensuales (para Vista 1 y 2) respetando restaurantes
sub_mes = df[(df[COL_FECHA] >= ini_mes) & (df[COL_FECHA] <= fin_mes)].copy()
sub_mes_ant = df[(df[COL_FECHA] >= ini_mes_ant) & (df[COL_FECHA] <= fin_mes_ant)].copy()

if rest_sel is not None:
    sub_mes = sub_mes[sub_mes[COL_CC].astype(str).isin(rest_sel)]
    sub_mes_ant = sub_mes_ant[sub_mes_ant[COL_CC].astype(str).isin(rest_sel)]

title_periodo = f"{mes_sel.strftime('%B %Y').title()} (1 al {fin_mes.day})"
st.markdown(f"## Ventas – {title_periodo}")

# ---------- KPIs de resumen (sobre el rango filtrado f) ----------
st.markdown("### Resumen general")

ventas_total = f[COL_VENTAS].sum()
n_tickets = f[COL_FOLIO].nunique() if COL_FOLIO in f.columns else None
aov = (ventas_total / n_tickets) if (ventas_total and n_tickets) else None

# Comparativo vs periodo anterior
# Lógica:
# - 1 día: vs día anterior
# - 2–19 días: vs misma cantidad de días anteriores
# - >=20 días: vs mes completo anterior
var_periodo_anterior = None
tiene_periodo_prev = False

if f_ini is not None and f_fin is not None:
    days = (f_fin - f_ini).days + 1
    if days > 0:
        if days == 1:
            cur_start, cur_end = f_ini, f_fin
            prev_start = f_ini - pd.Timedelta(days=1)
            prev_end = prev_start
        elif days < 20:
            cur_start, cur_end = f_ini, f_fin
            delta = pd.Timedelta(days=days)
            prev_start = f_ini - delta
            prev_end = f_fin - delta
        else:
            # modo mensual
            cur_start, cur_end = ini_mes, fin_mes
            prev_start, prev_end = ini_mes_ant, fin_mes_ant

        ini_c, fin_c = cur_start.date(), cur_end.date()
        ini_p, fin_p = prev_start.date(), prev_end.date()

        base_cur = df[
            (df[COL_FECHA].dt.date >= ini_c) & (df[COL_FECHA].dt.date <= fin_c)
        ].copy()

        base_prev = df[
            (df[COL_FECHA].dt.date >= ini_p) & (df[COL_FECHA].dt.date <= fin_p)
        ].copy()

        if rest_sel is not None:
            base_cur = base_cur[base_cur[COL_CC].astype(str).isin(rest_sel)]
            base_prev = base_prev[base_prev[COL_CC].astype(str).isin(rest_sel)]

        v_act = base_cur[COL_VENTAS].sum()
        v_prev = base_prev[COL_VENTAS].sum()

        if v_prev > 0:
            var_periodo_anterior = (v_act / v_prev) - 1
            tiene_periodo_prev = True
        else:
            var_periodo_anterior = None
            tiene_periodo_prev = False

# % delivery sobre ventas totales en f
pct_delivery = None
if COL_TIPO in f.columns and ventas_total:
    tot_del = f.loc[f[COL_TIPO].map(is_delivery), COL_VENTAS].sum()
    pct_delivery = tot_del / ventas_total if ventas_total else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ventas Netas (rango)", fmt_money(ventas_total))
c2.metric("Tickets (folios únicos)", f"{n_tickets:,.0f}" if n_tickets is not None else "—")
c3.metric("Ticket Promedio", f"{aov:,.0f}" if aov is not None else "—")
if var_periodo_anterior is not None:
    c4.metric(
        "% vs periodo anterior",
        f"{(var_periodo_anterior * 100):.1f}%",
        help="Comparado contra el periodo inmediatamente anterior (mismo número de días).",
    )
else:
    c4.metric(
        "% vs periodo anterior",
        "N/A",
        help="No hay ventas en el periodo anterior para hacer el comparativo.",
    )

# ---------- Tabs de análisis ----------
tab_tend, tab_v1, tab_v2, tab_heat, tab_top, tab_prod = st.tabs(
    ["Tendencia", "Ventas por restaurante", "Delivery", "Heatmap D/H", "Top productos", "Ventas por producto"]
)

# ======================================
# TENDENCIA (por restaurante + Tienda vs Delivery)
# ======================================
with tab_tend:
    st.markdown(f"### Tendencia de ventas por restaurante (agrupado por {granularidad.lower()})")

    # --- Tendencia por restaurante (misma lógica que Tienda vs Delivery) ---
    g_rest = f.copy()
    g_rest = agregar_periodo(g_rest, granularidad, COL_FECHA)

    # quitamos nulos por seguridad
    g_rest = g_rest.dropna(subset=[COL_CC, COL_VENTAS])

    # 1) agregamos igual que en Tienda vs Delivery, pero por restaurante
    ser_rest = (
        g_rest.groupby(["periodo", COL_CC], as_index=False)[COL_VENTAS]
        .sum()
        .sort_values("periodo")
    )

    if ser_rest.empty:
        st.info("No hay datos para mostrar la tendencia por restaurante en el rango seleccionado.")
    else:
        # 2) calculamos el top 8 sobre la serie ya agregada (como con Tienda vs Delivery)
        tot_por_rest = (
            ser_rest.groupby(COL_CC, as_index=False)[COL_VENTAS]
            .sum()
            .sort_values(COL_VENTAS, ascending=False)
        )
        top_rest = tot_por_rest[COL_CC].head(8).tolist()

        ser_rest_plot = ser_rest[ser_rest[COL_CC].isin(top_rest)]

        ch_rest = alt.Chart(ser_rest_plot).mark_line(point=True).encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y(f"{COL_VENTAS}:Q", title="Ventas"),
            color=alt.Color(f"{COL_CC}:N", title="Restaurante"),
            tooltip=[
                alt.Tooltip("periodo:T", title="Periodo"),
                alt.Tooltip(f"{COL_CC}:N", title="Restaurante"),
                alt.Tooltip(f"{COL_VENTAS}:Q", title="Ventas", format=","),
            ],
        ).properties(height=320)

        st.altair_chart(ch_rest, use_container_width=True)

    # --- Tienda vs Delivery (esto lo dejamos igual) ---
    st.markdown(f"### Tienda vs Delivery (mismo rango, agrupado por {granularidad.lower()})")

    g_split = f.copy()
    g_split["is_delivery"] = g_split[COL_TIPO].map(is_delivery)
    g_split = agregar_periodo(g_split, granularidad, COL_FECHA)

    ser2 = (
        g_split.groupby(["periodo", "is_delivery"], as_index=False)[COL_VENTAS]
        .sum()
        .sort_values("periodo")
    )
    ser2["canal"] = np.where(ser2["is_delivery"], "Delivery", "Tienda")

    if not ser2.empty:
        ch_split = alt.Chart(ser2).mark_line(point=True).encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y(f"{COL_VENTAS}:Q", title="Ventas"),
            color=alt.Color("canal:N", title="Canal"),
            tooltip=[
                alt.Tooltip("periodo:T", title="Periodo"),
                alt.Tooltip("canal:N", title="Canal"),
                alt.Tooltip(f"{COL_VENTAS}:Q", title="Ventas", format=","),
            ],
        ).properties(height=320)

        st.altair_chart(ch_split, use_container_width=True)
    else:
        st.info("No hay datos suficientes para mostrar Tienda vs Delivery en este rango.")

# ======================================
# VISTA 1 — Ventas Netas por CC (mes vs mes anterior)
# ======================================
with tab_v1:
    st.subheader("Ventas Netas por Restaurante (Mes vs Mes anterior)")

    if sub_mes.empty:
        st.info("No hay ventas en el mes seleccionado dentro del dataset (para los filtros actuales).")
    else:
        # Ventas mes por CC
        v_mes = (
            sub_mes.groupby(COL_CC, as_index=False)[COL_SUBTOT]
            .sum()
            .rename(columns={COL_SUBTOT: "Ventas Mes"})
        )
        # Ventas mes anterior por CC
        v_ant = (
            sub_mes_ant.groupby(COL_CC, as_index=False)[COL_SUBTOT]
            .sum()
            .rename(columns={COL_SUBTOT: "Ventas Mes Ant"})
        )

        vista1 = v_mes.merge(v_ant, on=COL_CC, how="left")

        # Budget = 5% arriba de Ventas Mes
        vista1["Budget"] = vista1["Ventas Mes"] * 1.05

        # % Alcance vs Budget
        vista1["% Alcance vs Budget"] = np.where(
            vista1["Budget"] > 0,
            vista1["Ventas Mes"] / vista1["Budget"],
            np.nan,
        )

        # % GAP vs Mes Anterior
        def calc_gap(vm, va):
            if pd.isna(va) or va == 0:
                return None
            return (vm / va) - 1

        vista1["% GAP vs Mes Ant"] = [
            calc_gap(vm, va) for vm, va in zip(vista1["Ventas Mes"], vista1["Ventas Mes Ant"])
        ]

        # Ordenar por Ventas Mes desc
        vista1 = vista1.sort_values("Ventas Mes", ascending=False)

        # Totales
        tot_vm = vista1["Ventas Mes"].sum()
        tot_va = vista1["Ventas Mes Ant"].sum()
        tot_budget = tot_vm * 1.05
        tot_alcance = (tot_vm / tot_budget) if tot_budget else np.nan
        tot_gap = calc_gap(tot_vm, tot_va)

        tot_row = pd.DataFrame(
            {
                COL_CC: ["Total Tiendas"],
                "Ventas Mes": [tot_vm],
                "Ventas Mes Ant": [tot_va],
                "Budget": [tot_budget],
                "% Alcance vs Budget": [tot_alcance],
                "% GAP vs Mes Ant": [tot_gap],
            }
        )

        vista1 = pd.concat([vista1, tot_row], ignore_index=True)

        # Solo tabla (quitamos la gráfica aquí)
        out1 = pd.DataFrame(
            {
                "Restaurante": vista1[COL_CC],
                "Ventas Netas (Mes)": vista1["Ventas Mes"].apply(fmt_money),
                "Budget (Mes)": vista1["Budget"].apply(fmt_money),
                "% Alcance Vs Budget": [
                    f"{fmt_pct(p)} {color_dot_for_target(p)}" for p in vista1["% Alcance vs Budget"]
                ],
                "% GAP vs Mes Anterior": [
                    f"{fmt_pct(p)} {arrow_gap(p)}" for p in vista1["% GAP vs Mes Ant"]
                ],
            }
        )
        st.dataframe(out1, use_container_width=True)

# ======================================
# VISTA 2 — Ventas Delivery (mes vs mes anterior)
# ======================================
with tab_v2:
    st.subheader("Ventas Delivery por Restaurante (Mes vs Mes anterior)")

    if sub_mes.empty:
        st.info("No hay ventas en el mes seleccionado dentro del dataset (para los filtros actuales).")
    else:
        sub_mes = sub_mes.copy()
        sub_mes_ant = sub_mes_ant.copy()

        sub_mes["__is_delivery"] = sub_mes[COL_TIPO].map(is_delivery)
        sub_mes_ant["__is_delivery"] = sub_mes_ant[COL_TIPO].map(is_delivery)

        # Ventas mes por CC (separando tienda vs delivery)
        tienda_mes = (
            sub_mes.loc[~sub_mes["__is_delivery"]]
            .groupby(COL_CC, as_index=False)[COL_SUBTOT]
            .sum()
            .rename(columns={COL_SUBTOT: "Ventas Tienda"})
        )
        delivery_mes = (
            sub_mes.loc[sub_mes["__is_delivery"]]
            .groupby(COL_CC, as_index=False)[COL_SUBTOT]
            .sum()
            .rename(columns={COL_SUBTOT: "Ventas Delivery"})
        )
        tot_mes = (
            sub_mes.groupby(COL_CC, as_index=False)[COL_SUBTOT]
            .sum()
            .rename(columns={COL_SUBTOT: "Ventas Totales"})
        )
        # Delivery mes anterior por CC
        delivery_ant = (
            sub_mes_ant.loc[sub_mes_ant["__is_delivery"]]
            .groupby(COL_CC, as_index=False)[COL_SUBTOT]
            .sum()
            .rename(columns={COL_SUBTOT: "Delivery Mes Ant"})
        )

        vista2 = (
            tot_mes.merge(tienda_mes, on=COL_CC, how="left")
            .merge(delivery_mes, on=COL_CC, how="left")
            .merge(delivery_ant, on=COL_CC, how="left")
        )

        vista2[["Ventas Tienda", "Ventas Delivery", "Ventas Totales", "Delivery Mes Ant"]] = (
            vista2[["Ventas Tienda", "Ventas Delivery", "Ventas Totales", "Delivery Mes Ant"]].fillna(0)
        )

        # % Tienda y % Delivery (complementarios respecto al total de ese CC)
        vista2["% Tienda"] = np.where(
            vista2["Ventas Totales"] > 0,
            vista2["Ventas Tienda"] / vista2["Ventas Totales"],
            np.nan,
        )
        vista2["% Delivery"] = np.where(
            vista2["Ventas Totales"] > 0,
            vista2["Ventas Delivery"] / vista2["Ventas Totales"],
            np.nan,
        )

        # % Mes Anterior (solo delivery)
        def pct_mes_ant(cur, ant):
            if ant == 0:
                return None
            return (cur / ant) - 1

        vista2["% Mes Anterior"] = [
            pct_mes_ant(c, a) for c, a in zip(vista2["Ventas Delivery"], vista2["Delivery Mes Ant"])
        ]

        # Orden sugerido: por Ventas Totales desc
        vista2 = vista2.sort_values("Ventas Totales", ascending=False)

        # Totales globales
        tot2_vm_tienda = vista2["Ventas Tienda"].sum()
        tot2_vm_deliv = vista2["Ventas Delivery"].sum()
        tot2_vm_total = vista2["Ventas Totales"].sum()
        tot2_vm_antdel = vista2["Delivery Mes Ant"].sum()
        tot2_pct_tienda = (tot2_vm_tienda / tot2_vm_total) if tot2_vm_total else None
        tot2_pct_delivery = (tot2_vm_deliv / tot2_vm_total) if tot2_vm_total else None
        tot2_pct_mes_ant = ((tot2_vm_deliv / tot2_vm_antdel) - 1) if tot2_vm_antdel else None

        tot2 = pd.DataFrame(
            [
                {
                    COL_CC: "Total Tiendas",
                    "Ventas Tienda": tot2_vm_tienda,
                    "Ventas Delivery": tot2_vm_deliv,
                    "Ventas Totales": tot2_vm_total,
                    "Delivery Mes Ant": tot2_vm_antdel,
                    "% Tienda": tot2_pct_tienda,
                    "% Delivery": tot2_pct_delivery,
                    "% Mes Anterior": tot2_pct_mes_ant,
                }
            ]
        )

        vista2 = pd.concat([vista2, tot2], ignore_index=True)

        # Presentación
        out2 = pd.DataFrame(
            {
                "Restaurante": vista2[COL_CC],
                "Ventas Netas Tienda": vista2["Ventas Tienda"].apply(fmt_money),
                "% Tienda": vista2["% Tienda"].apply(fmt_pct),
                "Ventas Delivery": vista2["Ventas Delivery"].apply(fmt_money),
                "% Delivery": vista2["% Delivery"].apply(fmt_pct),
                "Delivery Mes Anterior": vista2["Delivery Mes Ant"].apply(fmt_money),
                "% Mes Anterior": [
                    f"{fmt_pct(p)} {arrow_gap(p)}" for p in vista2["% Mes Anterior"]
                ],
            }
        )
        st.dataframe(out2, use_container_width=True)

# ======================================
# HEATMAP Día–Hora (en base al rango filtrado)
# ======================================
with tab_heat:
    st.subheader("Heatmap Día–Hora (rango filtrado)")

    g = f.copy()
    g["dow"] = g[COL_FECHA].dt.day_name()
    g["hora"] = g[COL_FECHA].dt.hour
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g["dow"] = pd.Categorical(g["dow"], categories=order, ordered=True)

    hm = g.groupby(["dow", "hora"], as_index=False)[COL_VENTAS].sum()
    chart = alt.Chart(hm).mark_rect().encode(
        x=alt.X("dow:O", title="Día"),
        y=alt.Y("hora:O", title="Hora"),
        color=alt.Color(f"{COL_VENTAS}:Q", title="Ventas"),
        tooltip=["dow:O", "hora:O", alt.Tooltip(f"{COL_VENTAS}:Q", format=",")],
    ).properties(height=320)

    st.altair_chart(chart, use_container_width=True)

# ======================================
# TOP PRODUCTOS / COMPLEMENTOS (rango filtrado)
# ======================================
def parse_detalle_items(texto: str):
    """
    Recibe el contenido de 'Detalle Items' y regresa:
    - lista de productos como tuplas (nombre, cantidad)
    - lista de complementos como tuplas (nombre, cantidad)

    Ejemplo:
    'Flauta de Pollo x2($79)' -> productos: [("Flauta de Pollo", 2)]
    """

    def normaliza_item(raw: str):
        txt = raw.strip()
        if not txt:
            return "", 0

        # Quitar precio tipo ($79) o ($79.00)
        txt = re.sub(r"\(\s*\$?\s*[\d\.,]+\s*\)", "", txt).strip()

        # Buscar patrón final " x2", " x10", etc.
        qty = 1
        m = re.search(r"\s[xX](\d+)\s*$", txt)
        if m:
            qty = int(m.group(1))
            txt = txt[:m.start()].rstrip()

        # Si quedó vacío, no contamos
        if not txt:
            return "", 0

        return txt, qty

    if not isinstance(texto, str) or not texto.strip():
        return [], []

    partes = [p.strip() for p in texto.split("|") if p.strip()]
    productos = []
    complems = []

    for p in partes:
        if "[" in p and "]" in p:
            # Ej: Combo x1 (...) [+Fresca, Grandes con Queso]
            try:
                left, right = p.split("[", 1)
                contenido = right.split("]", 1)[0]

                # Producto base
                base_name, base_qty = normaliza_item(left)
                if base_name and base_qty > 0:
                    productos.append((base_name, base_qty))

                # Complementos (normalmente sin xN, pero por si acaso se parsean igual)
                for c in contenido.split(","):
                    cname, cqty = normaliza_item(c)
                    if cname and cqty > 0:
                        complems.append((cname, cqty))
            except Exception:
                name, qty = normaliza_item(p)
                if name and qty > 0:
                    productos.append((name, qty))
        else:
            name, qty = normaliza_item(p)
            if name and qty > 0:
                productos.append((name, qty))

    return productos, complems


def parse_detalle_items_con_valor(texto: str):
    """
    Versión para 'Ventas por producto':
    - Solo considera productos base (no complementos).
    - Devuelve lista de tuplas: (nombre_producto, cantidad, precio_unitario)
      donde precio_unitario se calcula como: precio_renglón / cantidad.
    """
    def normaliza_item_con_precio(raw: str):
        txt = raw.strip()
        if not txt:
            return "", 0, None

        # Extraer precio tipo ($199.00)
        m_precio = re.search(r"\(\s*\$?\s*([\d\.,]+)\s*\)", txt)
        precio_linea = None
        if m_precio:
            num = m_precio.group(1).replace(",", "")
            try:
                precio_linea = float(num)
            except ValueError:
                precio_linea = None
            # Quitamos la parte del precio del texto
            txt = re.sub(r"\(\s*\$?\s*[\d\.,]+\s*\)", "", txt).strip()

        # Buscar patrón final " x2", " x10", etc.
        qty = 1
        m_qty = re.search(r"\s[xX](\d+)\s*$", txt)
        if m_qty:
            qty = int(m_qty.group(1))
            txt = txt[:m_qty.start()].rstrip()

        nombre = txt.strip()
        if not nombre:
            return "", 0, None

        precio_unit = None
        if precio_linea is not None and qty > 0:
            precio_unit = precio_linea / qty

        return nombre, qty, precio_unit

    productos = []

    if not isinstance(texto, str) or not texto.strip():
        return productos

    partes = [p.strip() for p in texto.split("|") if p.strip()]

    for p in partes:
        # Si trae complementos en corchetes, solo tomamos la parte base
        if "[" in p and "]" in p:
            left, _ = p.split("[", 1)
            base = left.strip()
        else:
            base = p

        nombre, qty, precio_unit = normaliza_item_con_precio(base)
        if nombre and qty > 0:
            productos.append((nombre, qty, precio_unit))

    return productos


with tab_top:
    st.subheader("Top 10 productos y complementos (por cantidad, rango filtrado)")

    if COL_DETALLE not in f.columns:
        st.info("No existe la columna 'Detalle Items' en el dataset.")
    else:
        registros_productos = []
        registros_complems = []

        for _, row in f.iterrows():
            detalle = row[COL_DETALLE]
            prods, comps = parse_detalle_items(detalle)

            # prods y comps vienen como (nombre, cantidad)
            for nombre, qty in prods:
                registros_productos.append({"item": nombre, "qty": qty})
            for nombre, qty in comps:
                registros_complems.append({"item": nombre, "qty": qty})

        df_prod = pd.DataFrame(registros_productos)
        df_comp = pd.DataFrame(registros_complems)

        modo = st.radio(
            "Ver:",
            ["Productos", "Complementos"],
            index=0,
            horizontal=True,
        )

        if modo == "Productos":
            if df_prod.empty:
                st.info("No se encontraron productos en el rango filtrado.")
            else:
                top_p = (
                    df_prod.groupby("item", as_index=False)["qty"]
                    .sum()
                    .sort_values("qty", ascending=False)
                    .head(10)
                )
                top_p = top_p.rename(columns={"qty": "cantidad"})

                ch_p = alt.Chart(top_p).mark_bar().encode(
                    x=alt.X("cantidad:Q", title="Cantidad vendida"),
                    y=alt.Y("item:N", sort="-x", title="Producto"),
                    tooltip=[
                        alt.Tooltip("item:N", title="Producto"),
                        alt.Tooltip("cantidad:Q", title="Cantidad"),
                    ],
                ).properties(height=400)

                st.altair_chart(ch_p, use_container_width=True)
                st.dataframe(top_p, use_container_width=True)
        else:
            if df_comp.empty:
                st.info("No se encontraron complementos en el rango filtrado.")
            else:
                top_c = (
                    df_comp.groupby("item", as_index=False)["qty"]
                    .sum()
                    .sort_values("qty", ascending=False)
                    .head(10)
                )
                top_c = top_c.rename(columns={"qty": "cantidad"})

                ch_c = alt.Chart(top_c).mark_bar().encode(
                    x=alt.X("cantidad:Q", title="Cantidad agregada"),
                    y=alt.Y("item:N", sort="-x", title="Complemento"),
                    tooltip=[
                        alt.Tooltip("item:N", title="Complemento"),
                        alt.Tooltip("cantidad:Q", title="Cantidad"),
                    ],
                ).properties(height=400)

                st.altair_chart(ch_c, use_container_width=True)
                st.dataframe(top_c, use_container_width=True)

with tab_prod:
    st.subheader("Ventas por producto (rango filtrado)")

    if COL_DETALLE not in f.columns:
        st.info("No existe la columna 'Detalle Items' en el dataset.")
    else:
        registros = []

        # Recorremos SOLO el df filtrado f (respeta todos los filtros)
        for _, row in f.iterrows():
            detalle = row.get(COL_DETALLE, "")
            productos = parse_detalle_items_con_valor(detalle)

            for nombre, qty, precio_unit in productos:
                registros.append(
                    {
                        "item": nombre,
                        "qty": qty,
                        "precio_unitario": precio_unit,
                    }
                )

        if not registros:
            st.info("No se encontraron productos en el rango filtrado.")
        else:
            df_prod_val = pd.DataFrame(registros)

            # Agregamos por producto
            df_resumen = (
                df_prod_val
                .groupby("item", as_index=False)
                .agg(
                    conteo=("qty", "sum"),
                    precio_promedio=("precio_unitario", "mean"),
                )
            )

            df_resumen["ventas_estimadas"] = (
                    df_resumen["conteo"] * df_resumen["precio_promedio"]
            )

            # Orden descendente por cantidad vendida
            df_resumen = df_resumen.sort_values(
                "conteo", ascending=False
            )

            st.dataframe(
                df_resumen.style.format(
                    {
                        "conteo": "{:,.0f}",
                        "precio_promedio": "${:,.2f}",
                        "ventas_estimadas": "${:,.2f}",
                    }
                ),
                use_container_width=True,
            )

st.caption(
    "Notas: 1) El comparativo vs periodo anterior usa: 1 día vs día anterior; 2–19 días vs misma cantidad de días previos; "
    "20+ días vs mes completo anterior. "
    "2) Budget temporal = 1.05 × Ventas Mes. "
    "3) Delivery se detecta si 'Tipo' contiene 'delivery'. "
    "4) La granularidad solo afecta las gráficas de tendencia, no los filtros. "
    "5) Todos los cálculos respetan los filtros de fechas y restaurantes."
)
