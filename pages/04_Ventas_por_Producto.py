# pages/04_Ventas_por_Producto.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
import re
import unicodedata

# ============= CONFIG BÁSICA =============
st.set_page_config(
    page_title="Ventas por Producto – Marcas HP",
    page_icon="🧾",
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
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .stDataFrame {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 0.3rem 0.3rem 0.8rem 0.3rem;
        box-shadow:0 14px 32px rgba(15, 23, 42, 0.12);
        border:1px solid rgba(148, 163, 184, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# URLs
# =========================================================
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"

# =========================================================
# Helpers
# =========================================================
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
        g["periodo"] = g[col_fecha].dt.to_period("W-MON").apply(lambda r: r.start_time)
    else:  # Mes
        g["periodo"] = g[col_fecha].dt.to_period("M").dt.to_timestamp()
    return g

def clean_money_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

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

def get_void_mask(df_: pd.DataFrame, col_estado: str) -> pd.Series:
    if col_estado in df_.columns:
        return df_[col_estado].astype(str).str.strip().str.lower().eq("void")
    return pd.Series(False, index=df_.index)

# =========================================================
# Parser (idéntico al tuyo)
# =========================================================
def _parse_base_item(raw: str):
    txt = raw.strip()
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
    m_qty = re.search(r"\s+[xX]\s*([\d\.]+)\s*$", txt)
    if m_qty:
        try:
            qty_float = float(m_qty.group(1))
            if qty_float < 1:
                qty = 1
            else:
                qty = int(round(qty_float))
        except ValueError:
            qty = 1
        txt = txt[:m_qty.start()].strip()

    nombre = txt.strip()
    if not nombre:
        return "", 0, None

    precio_unit = None
    if precio_linea is not None and qty > 0:
        precio_unit = precio_linea / qty

    return nombre, qty, precio_unit

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

        nombre_base, qty_base, precio_unit = _parse_base_item(base_texto)
        if qty_base <= 0:
            qty_base = 1

        if nombre_base:
            registros.append({
                "item": nombre_base,
                "qty": qty_base,
                "precio_unitario": precio_unit,
                "tipo_concepto": "base",
            })

        if complementos_texto:
            complementos = [c.strip() for c in complementos_texto.split(",") if c.strip()]
            for comp in complementos:
                comp_limpio = comp.lstrip("+").strip()
                if comp_limpio:
                    registros.append({
                        "item": comp_limpio,
                        "qty": qty_base,
                        "precio_unitario": None,
                        "tipo_concepto": "complemento",
                    })

    return registros

# =========================================================
# Carga de datos y catálogo
# =========================================================
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_URL)
    df_.columns = [c.strip() for c in df_.columns]
    return df_

@st.cache_data(ttl=600)
def load_catalogo() -> pd.DataFrame | None:
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
    cat["Clasificación"] = cat["Categoria_raw"]
    cat["es_remap"] = is_instr

    cat["concepto_key"] = cat["concepto"].map(norm_key)
    cat["canon_key"] = cat["concepto_canonico"].map(norm_key)
    return cat

# =========================================================
# Construcción de tabla flat con canonización (cache)
# =========================================================
@st.cache_data(ttl=600)
def build_items_flat(df: pd.DataFrame, catalogo: pd.DataFrame | None) -> pd.DataFrame:
    COL_CC = "Restaurante"
    COL_ESTADO = "Estado"
    COL_FECHA = "Fecha"
    COL_FOLIO = "Folio"
    COL_DETALLE = "Detalle Items"

    g = df.copy()
    g[COL_FECHA] = pd.to_datetime(g[COL_FECHA], errors="coerce", dayfirst=True)
    g = g[g[COL_FECHA].notna()].copy()

    # Flatten por fila (ticket)
    rows = []
    for _, r in g.iterrows():
        detalle = r.get(COL_DETALLE, "")
        regs = parse_detalle_items_base_y_complementos(detalle)
        if not regs:
            continue
        for it in regs:
            rows.append({
                "Fecha": r[COL_FECHA],
                "Restaurante": r.get(COL_CC, None),
                "Estado": r.get(COL_ESTADO, None),
                "Folio": r.get(COL_FOLIO, None),
                "tipo_concepto": it["tipo_concepto"],
                "item_raw": it["item"],
                "qty": it["qty"],
            })

    if not rows:
        return pd.DataFrame(columns=["Fecha","Restaurante","Estado","Folio","tipo_concepto","item_raw","qty","item","Clasificación"])

    flat = pd.DataFrame(rows)
    flat["item_key"] = flat["item_raw"].map(norm_key)

    if catalogo is None or catalogo.empty:
        flat["item"] = flat["item_raw"].astype(str)
        flat["Clasificación"] = "Sin clasificación"
        return flat

    # Merge catálogo
    j = flat.merge(
        catalogo,
        left_on="item_key",
        right_on="concepto_key",
        how="left",
        suffixes=("", "_cat"),
    )

    # Canonización por "Contar como"
    j["item"] = j["concepto_canonico"].fillna(j["item_raw"]).astype(str).str.strip()

    # Clasificación final para remap: buscar clasificación del concepto canónico (pero usando norm_key para que no falle por guiones/acentos)
    clas_base = (
        catalogo[~catalogo["es_remap"]]
        [["concepto","Categoria_raw"]]
        .drop_duplicates()
        .copy()
    )
    clas_base["concepto_key2"] = clas_base["concepto"].map(norm_key)
    clas_map = clas_base.set_index("concepto_key2")["Categoria_raw"].to_dict()

    j["Clasificación"] = j["Clasificación"].fillna(j["Categoria_raw"])
    j["Clasificación"] = j["Clasificación"].fillna("Sin clasificación").astype(str).str.strip()

    # Si es remap, usar la clasificación del canónico (por key)
    j["item_key2"] = j["item"].map(norm_key)
    mask_remap = j["es_remap"].fillna(False).astype(bool)
    j.loc[mask_remap, "Clasificación"] = j.loc[mask_remap, "item_key2"].map(lambda k: clas_map.get(k, "Sin clasificación"))

    # Filtrar "No contar"
    j = j[j["Clasificación"].str.strip().str.lower() != "no contar"].copy()

    keep = ["Fecha","Restaurante","Estado","Folio","tipo_concepto","item_raw","item","qty","Clasificación"]
    return j[keep]

# =========================================================
# UI
# =========================================================
st.sidebar.markdown("### Actualización")
if st.sidebar.button("🔄 Actualizar data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Última vista: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

df = load_data()
catalogo = load_catalogo()
flat = build_items_flat(df, catalogo)

st.markdown("## Ventas por Producto (conteo)")

if flat.empty:
    st.info("No hay items para analizar (revisa Detalle Items / rango de fechas).")
    st.stop()

# Filtros generales
min_f = flat["Fecha"].min().date()
max_f = flat["Fecha"].max().date()

c1, c2, c3 = st.columns([2,2,2])
with c1:
    rango = st.date_input("Rango de fechas", value=(min_f, max_f))
with c2:
    granularidad = st.radio("Periodo", ["Día","Semana","Mes"], index=2, horizontal=True)
with c3:
    incluir_void = st.toggle("Incluir VOID en conteo", value=False)

if isinstance(rango, (list, tuple)) and len(rango) == 2:
    f_ini = pd.to_datetime(rango[0])
    f_fin = pd.to_datetime(rango[1])
else:
    f_ini = pd.to_datetime(rango)
    f_fin = pd.to_datetime(rango)

mask = (flat["Fecha"].dt.date >= f_ini.date()) & (flat["Fecha"].dt.date <= f_fin.date())
flat_f = flat[mask].copy()

if not incluir_void:
    is_void = flat_f["Estado"].astype(str).str.strip().str.lower().eq("void")
    flat_f = flat_f[~is_void].copy()

# Filtro restaurante (opcional)
rests = sorted([x for x in flat_f["Restaurante"].dropna().unique().tolist()])
sel_rests = st.multiselect("Restaurantes", options=rests, default=rests)
if sel_rests:
    flat_f = flat_f[flat_f["Restaurante"].isin(sel_rests)].copy()

# Selector de productos (canónicos)
items = sorted([x for x in flat_f["item"].dropna().unique().tolist()])
default_pick = items[:1] if items else []
sel_items = st.multiselect("Productos (canónicos)", options=items, default=default_pick)

if not sel_items:
    st.info("Selecciona al menos 1 producto.")
    st.stop()

# Preparar serie
flat_f = flat_f[flat_f["item"].isin(sel_items)].copy()
flat_f = agregar_periodo(flat_f, granularidad, "Fecha")

serie = (
    flat_f.groupby(["periodo","item"], as_index=False)
    .agg(conteo=("qty","sum"))
    .sort_values(["periodo","item"])
)

st.markdown("### Conteo en el tiempo")

serie = serie.sort_values("periodo").copy()

if granularidad == "Día":
    serie["periodo_str"] = pd.to_datetime(serie["periodo"]).dt.strftime("%d %b %Y")
elif granularidad == "Semana":
    serie["periodo_str"] = pd.to_datetime(serie["periodo"]).dt.strftime("Sem %d %b %Y")
else:
    serie["periodo_str"] = pd.to_datetime(serie["periodo"]).dt.strftime("%b %Y")

orden_periodos = (
    serie[["periodo", "periodo_str"]]
    .drop_duplicates()
    .sort_values("periodo")["periodo_str"]
    .tolist()
)

base = (
    alt.Chart(serie)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X(
            "periodo_str:N",
            title=None,
            sort=orden_periodos,
            axis=alt.Axis(labelAngle=-45, labelLimit=140, ticks=False),
        ),
        y=alt.Y("conteo:Q", title="Unidades", scale=alt.Scale(zero=True)),
        tooltip=[
            alt.Tooltip("periodo_str:N", title=granularidad),
            alt.Tooltip("item:N", title="Producto"),
            alt.Tooltip("conteo:Q", title="Unidades", format=",.0f"),
        ],
    )
    .properties(height=140)  # ✅ el height va aquí
)

ch = (
    base.facet(
        row=alt.Row("item:N", title=None, header=alt.Header(labelAngle=0, labelPadding=6))
    )
    .configure_view(stroke=None)
    .configure_facet(spacing=8)
)

st.altair_chart(ch, use_container_width=True)


# Tabla pivote (opcional)
st.markdown("### Tabla (periodo × producto)")
pivot = serie.pivot_table(index="periodo", columns="item", values="conteo", aggfunc="sum", fill_value=0).reset_index()
st.dataframe(pivot, use_container_width=True)

# Top restaurantes para el/los productos (útil cuando seleccionas varios)
st.markdown("### Top restaurantes (en el rango)")
top_rest = (
    flat_f.groupby(["Restaurante","item"], as_index=False)
    .agg(conteo=("qty","sum"))
    .sort_values(["conteo"], ascending=False)
)
st.dataframe(top_rest, use_container_width=True)
