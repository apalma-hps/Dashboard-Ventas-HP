# pages/02_Health_Rate_por_Restaurante.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
import re
import unicodedata

# ============= CONFIG BÁSICA =============
st.set_page_config(
    page_title="Health Rate por Restaurante – Marcas HP",
    page_icon="📊",
    layout="wide",
)

# ===== Tema de Altair (mismo que en app.py) =====
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
        <h1 style="margin-bottom:0;">Ventas y Health Rate – Marcas HP</h1>
        <p style="color:#6F7277;font-size:0.95rem;margin-top:0.25rem;">
        Dashboard ejecutivo · Rendimiento por restaurante
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================================
# URLs
# =========================================================

DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQZBL6mvIC1OUC-p0MREMW_7UvMKb8It4Y_ldFOi3FbqP4cwZBLrDXwpA_hjBzkeZz3tsOBqd9BlamY/pub?output=csv"

CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"

# =========================================================
# Helpers (formato / periodos)
# =========================================================

def fmt_money(x):
    return "—" if pd.isna(x) else f"${x:,.0f}"

def fmt_pct(x):
    return "—" if (x is None or pd.isna(x)) else f"{x * 100:,.0f}%"

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

# =========================================================
# Normalización robusta para match (aplanado + vinculación)
# =========================================================

def norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()

    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

    # conserva / y -
    s = re.sub(r"[^\w\s\/\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\-+", "-", s).strip()
    return s

# =========================================================
# Parser (base + complementos)
# =========================================================

def _parse_base_item(raw: str):
    txt = raw.strip()
    if not txt:
        return "", 0, None

    # precio línea
    m_precio = re.search(r"\(\s*\$?\s*([\d\.,]+)\s*\)", txt)
    precio_linea = None
    if m_precio:
        num = m_precio.group(1).replace(",", "")
        try:
            precio_linea = float(num)
        except ValueError:
            precio_linea = None
        txt = re.sub(r"\(\s*\$?\s*[\d\.,]+\s*\)", "", txt).strip()

    # qty xN
    qty = 1
    m_qty_int = re.search(r"\s[xX]\s*(\d+)\s*$", txt)
    if m_qty_int:
        qty = int(m_qty_int.group(1))
        txt = txt[:m_qty_int.start()].rstrip()
    else:
        # limpiar x0.5 etc.
        m_qty_float = re.search(r"\s[xX]\s*([\d\.,]+)\s*$", txt)
        if m_qty_float:
            txt = txt[:m_qty_float.start()].rstrip()

    nombre = txt.strip()
    if not nombre:
        return "", 0, None

    precio_unit = None
    if precio_linea is not None and qty > 0:
        precio_unit = precio_linea / qty

    return nombre, qty, precio_unit

def parse_detalle_items_base_y_complementos(texto: str):
    """
    Devuelve lista de dicts:
      item, qty, precio_unitario (solo base), tipo_concepto (base/complemento)
    Complementos: se cuentan como 1 por aparición (según tu regla actual).
    """
    registros = []
    if not isinstance(texto, str) or not texto.strip():
        return registros

    partes = [p.strip() for p in texto.split("|") if p.strip()]

    for p in partes:
        complemento_texto = None
        base_texto = p

        if "[" in p and "]" in p:
            left, right = p.split("[", 1)
            base_texto = left.strip()
            complemento_texto = right.rsplit("]", 1)[0].strip()

        # Base
        nombre_base, qty_base, precio_unit = _parse_base_item(base_texto)
        if nombre_base and qty_base > 0:
            registros.append(
                {
                    "item": nombre_base,
                    "qty": qty_base,
                    "precio_unitario": precio_unit,
                    "tipo_concepto": "base",
                }
            )

        # Complementos
        if complemento_texto:
            comps_raw = [c.strip() for c in complemento_texto.split(",") if c.strip()]
            for c in comps_raw:
                if c.startswith("+"):
                    c = c[1:].strip()
                if not c:
                    continue
                registros.append(
                    {
                        "item": c,
                        "qty": 1,
                        "precio_unitario": None,
                        "tipo_concepto": "complemento",
                    }
                )

    return registros

# =========================================================
# Carga de datos y catálogo
# =========================================================

@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(ttl=600)
def load_catalogo() -> pd.DataFrame | None:
    """
    Catálogo con columnas:
      - concepto
      - tipo_concepto
      - conteo_total (no se usa para cálculo del dashboard)
      - Categoria (incluye No contar / Contar como X / etc.)
    """
    try:
        cat = pd.read_csv(CATALOGO_URL)
    except Exception as e:
        st.warning(f"No se pudo cargar el catálogo: {e}")
        return None

    cat.columns = [c.strip() for c in cat.columns]
    required = {"concepto", "tipo_concepto", "Categoria"}
    if not required.issubset(set(cat.columns)):
        st.warning("El catálogo debe tener columnas: concepto, tipo_concepto, Categoria (conteo_total es opcional).")
        return None

    cat["concepto"] = cat["concepto"].astype(str).str.strip()
    cat["tipo_concepto"] = cat["tipo_concepto"].astype(str).str.strip().str.lower()
    cat["Categoria_raw"] = cat["Categoria"].astype(str).str.strip()

    # Instrucción: "Contar como X" o "Contar X"
    is_instr = cat["Categoria_raw"].str.match(r"(?i)^\s*contar\s+")
    m = cat["Categoria_raw"].str.extract(r"(?i)^\s*contar\s*(?:como\s+)?(.+?)\s*$")

    cat["concepto_canonico"] = np.where(is_instr, m[0].str.strip(), cat["concepto"])
    cat["Clasificación"] = np.where(is_instr, "REMAP", cat["Categoria_raw"])

    # Llaves para merge robusto
    cat["concepto_key"] = cat["concepto"].map(norm_key)
    cat["canon_key"] = cat["concepto_canonico"].map(norm_key)

    return cat

# =========================================================
# Preparación de DF principal
# =========================================================

df = load_data()
catalogo = load_catalogo()

COL_CC = "Restaurante"
COL_FECHA = "Fecha"
COL_SUBTOT = "Subtotal"
COL_TIPO = "Tipo"
COL_FOLIO = "Folio"
COL_DETALLE = "Detalle Items"
COL_VENTAS = COL_SUBTOT

df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)
df[COL_SUBTOT] = (
    df[COL_SUBTOT]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# =========================================================
# FILTROS
# =========================================================

st.sidebar.markdown("### Filtros")

if df[COL_FECHA].notna().any():
    min_f = df[COL_FECHA].min()
    max_f = df[COL_FECHA].max()
    rango = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_f.date(), max_f.date()),
    )

    if isinstance(rango, (list, tuple)):
        if len(rango) == 2:
            f_ini, f_fin = [pd.to_datetime(x) for x in rango]
        else:
            f_ini = f_fin = pd.to_datetime(rango[0])
    else:
        f_ini = f_fin = pd.to_datetime(rango)
else:
    f_ini = f_fin = None

granularidad = st.sidebar.radio(
    "Ver periodo como",
    ["Día", "Semana", "Mes"],
    index=2,
    horizontal=True,
)

df_filt = df.copy()
if f_ini is not None and f_fin is not None:
    ini_d, fin_d = f_ini.date(), f_fin.date()
    mask = (df_filt[COL_FECHA].dt.date >= ini_d) & (df_filt[COL_FECHA].dt.date <= fin_d)
    df_filt = df_filt[mask]

if df_filt.empty:
    st.info("No hay datos en el rango de fechas seleccionado.")
    st.stop()

rests = sorted(df_filt[COL_CC].dropna().unique().tolist())
tabs = st.tabs(rests)

# =========================================================
# CONTENIDO POR RESTAURANTE
# =========================================================

for rest_name, tab in zip(rests, tabs):
    with tab:
        st.markdown(f"### {rest_name}")

        data_rest = df_filt[df_filt[COL_CC] == rest_name].copy()
        if data_rest.empty:
            st.info("Sin datos para este restaurante en el rango seleccionado.")
            continue

        # --- KPIs (sin cambios) ---
        ventas_total = data_rest[COL_VENTAS].sum()

        n_tickets = data_rest[COL_FOLIO].nunique() if COL_FOLIO in data_rest.columns else None
        ticket_prom = (ventas_total / n_tickets) if n_tickets and n_tickets > 0 else None

        dias_unicos = data_rest[COL_FECHA].dt.date.nunique()
        prom_diario_ventas = (ventas_total / dias_unicos) if (ventas_total and dias_unicos) else None

        aport_delivery = None
        if COL_TIPO in data_rest.columns and ventas_total:
            tot_del = data_rest.loc[data_rest[COL_TIPO].map(is_delivery), COL_VENTAS].sum()
            aport_delivery = tot_del / ventas_total if ventas_total else None

        kpi_rows = [
            {"KPI": "Ventas Netas (rango)", "Valor": fmt_money(ventas_total)},
            {"KPI": "Ticket Promedio", "Valor": fmt_money(ticket_prom) if ticket_prom is not None else "—"},
            {"KPI": "Promedio Diario de Ventas", "Valor": fmt_money(prom_diario_ventas) if prom_diario_ventas is not None else "—"},
            {"KPI": "Aportación Delivery", "Valor": fmt_pct(aport_delivery)},
        ]
        st.dataframe(pd.DataFrame(kpi_rows).set_index("KPI"), use_container_width=True)

        # --- Gráfica tiempo (sin cambios) ---
        data_rest_period = agregar_periodo(data_rest, granularidad, COL_FECHA)
        serie = (
            data_rest_period.groupby("periodo", as_index=False)[COL_VENTAS]
            .sum()
            .sort_values("periodo")
        )

        st.markdown("#### Ventas en el tiempo")
        ch = (
            alt.Chart(serie)
            .mark_line(point=True)
            .encode(
                x=alt.X("periodo:T", title=granularidad),
                y=alt.Y(f"{COL_VENTAS}:Q", title="Ventas netas"),
                tooltip=[
                    alt.Tooltip("periodo:T", title=granularidad),
                    alt.Tooltip(f"{COL_VENTAS}:Q", title="Ventas", format=","),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(ch, use_container_width=True)

        # =========================================================
        # Conteo por concepto (base + complementos) + Clasificación
        # =========================================================
        st.markdown("#### Ventas Estimadas por Concepto)

        if COL_DETALLE not in data_rest.columns:
            st.info("No existe la columna 'Detalle Items' en la base de datos.")
            continue

        registros = []
        for _, row in data_rest.iterrows():
            detalle = row.get(COL_DETALLE, "")
            registros.extend(parse_detalle_items_base_y_complementos(detalle))

        if not registros:
            st.info("No se encontraron conceptos en el rango filtrado para este restaurante.")
            continue

        df_flat = pd.DataFrame(registros)
        df_flat["item_key"] = df_flat["item"].map(norm_key)

        if catalogo is None:
            # fallback sin catálogo
            df_resumen = (
                df_flat.groupby(["tipo_concepto", "item"], as_index=False)
                .agg(
                    conteo=("qty", "sum"),
                    precio_promedio=("precio_unitario", "mean"),
                )
            )
            df_resumen["ventas_estimadas"] = (df_resumen["conteo"] * df_resumen["precio_promedio"]).fillna(0)
            df_resumen = df_resumen.sort_values(["tipo_concepto", "conteo"], ascending=[True, False])

            st.dataframe(
                df_resumen.style.format(
                    {"conteo": "{:,.0f}", "precio_promedio": "${:,.2f}", "ventas_estimadas": "${:,.2f}"}
                ),
                use_container_width=True,
            )
            continue

        # --- Merge robusto ---
        df_join = df_flat.merge(
            catalogo,
            left_on="item_key",
            right_on="concepto_key",
            how="left",
            suffixes=("", "_cat"),
        )

        # --- Alias "Contar como X" / "Contar X" ---
        df_join["item_canonico"] = df_join["concepto_canonico"].fillna(df_join["item"]).astype(str).str.strip()

        # --- Clasificación para tabs ---
        # Si no viene del catálogo => Sin clasificación
        df_join["Clasificación"] = df_join["Clasificación"].fillna("Sin clasificación").astype(str).str.strip()

        # --- Excluir No contar ---
        df_join = df_join[df_join["Clasificación"].str.strip().str.lower() != "no contar"]

        # --- Quitar filas REMAP (solo eran alias, ya se movieron al canónico) ---
        df_join = df_join[df_join["Clasificación"] != "REMAP"]

        # Auditoría de no mapeados
        no_mapeados = (
            df_join[df_join["concepto"].isna()][["item", "item_key", "tipo_concepto"]]
            .drop_duplicates()
            .sort_values(["tipo_concepto", "item"])
        )
        if not no_mapeados.empty:
            with st.expander("⚠️ Conceptos sin clasificar (actualiza el catálogo)"):
                st.dataframe(no_mapeados, use_container_width=True)

        # Precio promedio SOLO para bases (complementos no traen precio)
        df_join["precio_unitario_base"] = np.where(
            df_join["tipo_concepto"].eq("base"),
            df_join["precio_unitario"],
            np.nan,
        )

        # Agregación por Clasificación + concepto canónico
        df_resumen = (
            df_join
            .groupby(["Clasificación", "item_canonico"], as_index=False)
            .agg(
                conteo=("qty", "sum"),
                precio_promedio=("precio_unitario_base", "mean"),
            )
            .rename(columns={"item_canonico": "item"})
        )

        df_resumen["ventas_estimadas"] = (df_resumen["conteo"] * df_resumen["precio_promedio"]).fillna(0)
        df_resumen = df_resumen.sort_values(["Clasificación", "conteo"], ascending=[True, False])

        # Tabs por clasificación
        clasifs = df_resumen["Clasificación"].dropna().unique().tolist()
        clas_tabs = st.tabs(clasifs)

        for clas, t_clas in zip(clasifs, clas_tabs):
            with t_clas:
                st.markdown(f"##### {clas}")

                df_sub = (
                    df_resumen[df_resumen["Clasificación"] == clas]
                    .drop(columns=["Clasificación"])  # 👈 AQUÍ
                    .copy()
                )

                st.dataframe(
                    df_sub.style.format(
                        {
                            "conteo": "{:,.0f}",
                            "precio_promedio": "${:,.2f}",
                            "ventas_estimadas": "${:,.2f}",
                        }
                    ),
                    use_container_width=True,
                )

