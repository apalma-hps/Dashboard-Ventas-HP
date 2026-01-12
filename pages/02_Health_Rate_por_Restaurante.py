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
        <h1 style="margin-bottom:0;">Ventas y Health Rate – Marcas HP</h1>
        <p style="color:#6F7277;font-size:0.95rem;margin-top:0.25rem;">
        Rendimiento mensual por restaurante.
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================================
# URLs
# =========================================================
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSLIeswEs8OILxZmVMwObbli0Zpbbqx7g7h6ZC5Fwm0PCjlZEFy66L9Xpha6ROW3loFCIRiWvEnLRHS/pub?output=csv"
CATALOGO_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQtKQGyCaerGAedhlpzaXlr-ycmm1t08a6lUtg-_3f7yWtJhLkQ6vn0TlI89l0FGVxOUy1Cwj5ykliB/pub?output=csv"


# =========================================================
# Helpers (formato / periodos)
# =========================================================
def fmt_money(x):
    return "—" if (x is None or pd.isna(x)) else f"${x:,.0f}"


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
# Parser (base + complementos) - MEJORADO
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
    m_qty = re.search(r"\s+[xX]\s*(\d+)\s*$", txt)
    if m_qty:
        qty = int(m_qty.group(1))
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

        if nombre_base and qty_base > 0:
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
                        "qty": 1,
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
        st.warning("El catálogo debe tener columnas: concepto, tipo_concepto, Categoria (conteo_total es opcional).")
        return None

    cat["concepto"] = cat["concepto"].astype(str).str.strip()
    cat["tipo_concepto"] = cat["tipo_concepto"].astype(str).str.strip().str.lower()
    cat["Categoria_raw"] = cat["Categoria"].astype(str).str.strip()

    is_instr = cat["Categoria_raw"].str.match(r"(?i)^\s*contar\s+")
    m = cat["Categoria_raw"].str.extract(r"(?i)^\s*contar\s*(?:como\s+)?(.+?)\s*$")

    cat["concepto_canonico"] = np.where(is_instr, m[0].str.strip(), cat["concepto"])
    cat["Clasificación"] = np.where(is_instr, "REMAP", cat["Categoria_raw"])

    cat["concepto_key"] = cat["concepto"].map(norm_key)
    cat["canon_key"] = cat["concepto_canonico"].map(norm_key)

    return cat


# =========================================================
# Preparación de DF principal
# =========================================================
df = load_data()
catalogo = load_catalogo()

COL_CC = "Restaurante"
COL_ESTADO = "Estado"
COL_FECHA = "Fecha"
COL_SUBTOT = "Subtotal"
COL_TOTAL = "Total"
COL_DESCUENTOS = "Descuentos"
COL_TIPO = "Tipo"
COL_FOLIO = "Folio"
COL_DETALLE = "Detalle Items"

COL_VENTAS = "ventas_efectivas"

df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce", dayfirst=True)

if COL_TOTAL not in df.columns:
    st.error("No existe la columna 'Total' en la base de datos. Revisa el Google Sheet publicado.")
    st.stop()

df[COL_TOTAL] = clean_money_series(df[COL_TOTAL]).fillna(0.0)

if COL_SUBTOT not in df.columns:
    st.warning("No existe la columna 'Subtotal'. Se usará Subtotal = Total como fallback (puede afectar precisión).")
    df[COL_SUBTOT] = df[COL_TOTAL]
else:
    df[COL_SUBTOT] = clean_money_series(df[COL_SUBTOT]).fillna(0.0)

if COL_DESCUENTOS not in df.columns:
    st.warning("No existe la columna 'Descuentos'. Se asumirá descuento = 0.")
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
if not rests:
    st.info("No hay restaurantes disponibles para el rango seleccionado.")
    st.stop()

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

        is_void_r = get_void_mask(data_rest, COL_ESTADO)

        ventas_total = float(data_rest[COL_VENTAS].sum())
        descuentos_total = float(data_rest[COL_DESCUENTOS].sum())

        # ✅ Tickets = Folios NO VOID (no depende de ventas>0)
        n_tickets = int(data_rest.loc[~is_void_r, COL_FOLIO].nunique()) if COL_FOLIO in data_rest.columns else 0
        ticket_prom = (ventas_total / n_tickets) if n_tickets > 0 else None

        dias_unicos = int(data_rest[COL_FECHA].dt.date.nunique())
        prom_diario_ventas = (ventas_total / dias_unicos) if (dias_unicos > 0) else None

        aport_delivery = None
        if COL_TIPO in data_rest.columns and ventas_total > 0:
            tot_del = float(data_rest.loc[data_rest[COL_TIPO].map(is_delivery), COL_VENTAS].sum())
            aport_delivery = tot_del / ventas_total

        n_void = int(data_rest.loc[is_void_r, COL_FOLIO].nunique()) if COL_FOLIO in data_rest.columns else int(is_void_r.sum())

        kpi_rows = [
            {"KPI": "Ventas Totales (netas)", "Valor": fmt_money(ventas_total)},
            {"KPI": "Descuentos Totales", "Valor": fmt_money(descuentos_total)},
            {"KPI": "Tickets (No Void)", "Valor": f"{n_tickets:,}"},
            {"KPI": "Tickets Cancelados (Void)", "Valor": f"{n_void:,}"},
            {"KPI": "Ticket Promedio", "Valor": fmt_money(ticket_prom) if ticket_prom is not None else "—"},
            {"KPI": "Promedio Diario de Ventas", "Valor": fmt_money(prom_diario_ventas) if prom_diario_ventas is not None else "—"},
            {"KPI": "Aportación Delivery", "Valor": fmt_pct(aport_delivery) if aport_delivery is not None else "—"},
        ]
        st.dataframe(pd.DataFrame(kpi_rows).set_index("KPI"), use_container_width=True)

        # --- Gráfica tiempo ---
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
                y=alt.Y(f"{COL_VENTAS}:Q", title="Ventas"),
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
        st.markdown("#### Conteo por concepto (base + complementos) · solo este restaurante")

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

        df_join = df_flat.merge(
            catalogo,
            left_on="item_key",
            right_on="concepto_key",
            how="left",
            suffixes=("", "_cat"),
        )

        df_join["item_canonico"] = df_join["concepto_canonico"].fillna(df_join["item"]).astype(str).str.strip()
        df_join["Clasificación"] = df_join["Clasificación"].fillna("Sin clasificación").astype(str).str.strip()

        df_join = df_join[df_join["Clasificación"].str.strip().str.lower() != "no contar"]
        df_join = df_join[df_join["Clasificación"] != "REMAP"]

        no_mapeados = (
            df_join[df_join["concepto"].isna()][["item", "item_key", "tipo_concepto"]]
            .drop_duplicates()
            .sort_values(["tipo_concepto", "item"])
        )
        if not no_mapeados.empty:
            with st.expander("⚠️ Conceptos sin clasificar (actualiza el catálogo)"):
                st.dataframe(no_mapeados, use_container_width=True)

        df_join["precio_unitario_base"] = np.where(
            df_join["tipo_concepto"].eq("base"),
            df_join["precio_unitario"],
            np.nan,
        )

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

        clasifs = df_resumen["Clasificación"].dropna().unique().tolist()
        clas_tabs = st.tabs(clasifs)

        for clas, t_clas in zip(clasifs, clas_tabs):
            with t_clas:
                st.markdown(f"##### {clas}")

                df_sub = (
                    df_resumen[df_resumen["Clasificación"] == clas]
                    .drop(columns=["Clasificación"])
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
