
# Tienda Aurelion - App Interactiva (Streamlit)
# ------------------------------------------------
# Requisitos:
#   pip install streamlit pandas openpyxl
#
# Ejecutar:
#   streamlit run aurelion_app.py
#
# Estructura esperada:
#   AURELION/
#   ├── aurelion_ap.py
#   ├── BD/
#   │   ├── clientes.xlsx
#   │   ├── productos.xlsx
#   │   ├── ventas.xlsx
#   │   └── detalle_ventas.xlsx
#   └── IMAGES/
#       └── LOGO.png
#       └── LOGO2.png

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Tienda Aurelion - Analítica",
    page_icon="🛒",
    layout="wide"
)

# Ruta del logo
logo_path = Path(__file__).parent / "IMAGES" / "LOGO2.png"

# Encabezado con fondo degradado y columnas
col_logo, col_text = st.columns([1, 3])
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.write("🖼️ (Logo no encontrado)")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <h2 style="margin:6px 0 0 6px; opacity:.9; color:#73FF86">Comportamiento de clientes y métodos de pago</h2>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------
# Funciones auxiliares
# --------------------------------------
@st.cache_data(show_spinner=False)
def load_excel_or_prompt(default_path: Path, label: str) -> pd.DataFrame:
    if default_path.exists():
        return pd.read_excel(default_path)
    st.info(f"No se encontró **{default_path.name}**. Subilo para continuar.")
    file = st.file_uploader(f"Subir {label} ({default_path.name})", type=["xlsx"], key=f"uploader_{label}")
    if file is not None:
        return pd.read_excel(file)
    st.stop()

def dtype_to_scale(dtype: str, colname: str) -> str:
    d = dtype.lower()
    name = colname.lower()
    if any(k in name for k in ["id_", "email", "nombre", "ciudad", "categoria", "medio_pago", "id"]):
        if "id" in name:
            return "Nominal (identificador)"
        return "Nominal (categórica)"
    if "datetime" in d or "date" in d:
        return "Temporal (fecha/tiempo)"
    if "int" in d or "float" in d:
        if any(k in name for k in ["cantidad", "precio", "importe", "monto", "total"]):
            return "Razón (numérica)"
        return "Intervalo / Razón (numérica)"
    return "Nominal"

def schema_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "columna": df.columns,
        "dtype_pandas": [str(df[c].dtype) for c in df.columns],
        "escala_aprox": [dtype_to_scale(str(df[c].dtype), c) for c in df.columns]
    })

# --------------------------------------
# Rutas relativas
# --------------------------------------
base_path = Path(__file__).parent / "BD"

data_paths: Dict[str, Path] = {
    "clientes": base_path / "clientes.xlsx",
    "productos": base_path / "productos.xlsx",
    "ventas": base_path / "ventas.xlsx",
    "detalle_ventas": base_path / "detalle_ventas.xlsx",
}

with st.spinner("Cargando datasets..."):
    clientes = load_excel_or_prompt(data_paths["clientes"], "clientes")
    productos = load_excel_or_prompt(data_paths["productos"], "productos")
    ventas = load_excel_or_prompt(data_paths["ventas"], "ventas")
    detalle_ventas = load_excel_or_prompt(data_paths["detalle_ventas"], "detalle_ventas")


# --------------------------------------
# Navegación lateral (dividida por Sprints)
# --------------------------------------
st.sidebar.title("Navegación")

sprint = st.sidebar.radio(
    "Sprint:",
    ["SPRINT 1 — Fundamentos", "SPRINT 2 — Análisis"],
    key="sprint_sel"
)

if "SPRINT 1" in sprint:
    # Mantiene exactamente las secciones originales del Sprint 1
    section = st.sidebar.radio(
        "Secciones SPRINT 1:",
        ["Temas", "Fuentes", "Pseudocódigo", "Diagrama", "Resumen Sprint 1"],
        key="s1_sec"
    )
elif "SPRINT 2" in sprint:
    section = st.sidebar.radio(
        "Secciones SPRINT 2:",
        ["Análisis de datos",
         "Modelo de Datos",
         "Distribución de Variables",
         "Correlaciones entre variables principales",
          "RFM (Segmentación clientes)",
          "Resumen Sprint 2"
         ],   
        key="s2_sec"
    )



# --------------------------------------
# TEMAS
# --------------------------------------
if section == "Temas":
    st.subheader("Temas del TP")
    
    tema = st.selectbox(
        "Elegí un tema",
        ["Tema 1 — Comportamiento de clientes y fidelización",
         "Tema 2 — Preferencias de pago y su impacto"]
    )

    if "Tema 1" in tema:
        st.markdown("### 🧠 Tema 1 — Comportamiento de clientes y fidelización")
        st.markdown(
            """
            **Problema:** La tienda no tiene visibilidad clara sobre la **frecuencia de compra**, **antigüedad** y **actividad** de los clientes.

            **Solución propuesta:** Construir indicadores de **frecuencia de compra**, **recencia** (tiempo desde la última compra) y **monetización** (importe total), así como segmentaciones de clientes **activos / inactivos / nuevos**.

            **Posible aplicación:** Implementar campañas de **fidelización** y **promociones personalizadas** (descuentos, cupones, puntos) enfocadas en clientes con alta probabilidad de recompra o en riesgo de abandono.
            """
        )

        with st.expander("Vista rápida de datos relacionados (clientes + ventas)"):
            ventas_c = ventas.merge(clientes[["id_cliente", "ciudad", "fecha_alta"]], on="id_cliente", how="left")
            compras_por_cliente = ventas_c.groupby("id_cliente")["id_venta"].nunique().rename("compras").reset_index()
            top_clientes = compras_por_cliente.sort_values("compras", ascending=False).head(10)

            col1, col2 = st.columns(2)
            col1.metric("Clientes totales", int(clientes["id_cliente"].nunique()))
            col2.metric("Ventas totales", int(ventas["id_venta"].nunique()))
            st.write("**Top 10 por cantidad de compras**")
            st.dataframe(top_clientes, use_container_width=True, hide_index=True)

    if "Tema 2" in tema:
        st.markdown("### 💳 Tema 2 — Preferencias de pago y su impacto en las ventas")
        st.markdown(
            """
            **Problema:** Se desconoce si el **método de pago** (tarjeta, QR, transferencia, etc.) influye en el volumen de ventas y en qué periodos.

            **Solución propuesta:** Analizar la distribución de ventas por **medio de pago** y su **evolución temporal**; identificar picos, estacionalidad y oportunidades de **promociones específicas** por método de pago.

            **Posible aplicación:** Negociar **beneficios con proveedores de pago** (cashback, cuotas sin interés) y comunicar **promociones** en los días/horas de mayor adopción del método seleccionado.
            """
        )

        with st.expander("Vista rápida: Ventas por método de pago"):
            ventas_por_pago = ventas.groupby("medio_pago")["id_venta"].nunique().sort_values(ascending=False).reset_index()
            ventas_por_pago.columns = ["medio_pago", "ventas"]
            st.dataframe(ventas_por_pago, use_container_width=True, hide_index=True)

# --------------------------------------
# FUENTES
# --------------------------------------
if section == "Fuentes":
    st.subheader("Fuentes — Datasets de referencia")
    st.caption("**Fuente general:** Archivos provistos para el TP de Tienda Aurelion (datasets sintéticos).")

    with st.expander("📁 clientes.xlsx — Definición, estructura, tipos y escala"):
        st.markdown("**Definición:** Maestro de clientes con datos básicos de identificación y alta.")
        st.dataframe(schema_table(clientes), use_container_width=True, hide_index=True)
        st.dataframe(clientes.head(), use_container_width=True)

    with st.expander("📁 ventas.xlsx — Definición, estructura, tipos y escala"):
        st.markdown("**Definición:** Cabecera de ventas con la fecha, el cliente asociado y el método de pago.")
        st.dataframe(schema_table(ventas), use_container_width=True, hide_index=True)
        st.dataframe(ventas.head(), use_container_width=True)

    with st.expander("📁 detalle_ventas.xlsx — Definición, estructura, tipos y escala"):
        st.markdown("**Definición:** Detalle de cada venta con cantidades, precios e importes.")
        st.dataframe(schema_table(detalle_ventas), use_container_width=True, hide_index=True)
        st.dataframe(detalle_ventas.head(), use_container_width=True)

    with st.expander("📁 productos.xlsx — Definición, estructura, tipos y escala"):
        st.markdown("**Definición:** Catálogo de productos con su categoría y precio unitario.")
        st.dataframe(schema_table(productos), use_container_width=True, hide_index=True)
        st.dataframe(productos.head(), use_container_width=True)

st.markdown(
    """
    <hr style="margin: 32px 0; border: none; border-top: 1px solid rgba(120,120,120,.2)" />

    """,
    unsafe_allow_html=True
)

# --------------------------------------
# PSEUDOCÓDIGO
# --------------------------------------
if section == "Pseudocódigo":
    st.subheader("🧩 Pseudocódigo del Proyecto")

    st.markdown(
        """
        A continuación se presentan los pseudocódigos principales utilizados para
        la resolución de los temas del proyecto *Tienda Aurelion*.
        """
    )

    tema_pseudo = st.selectbox(
        "Elegí el tema para visualizar su pseudocódigo:",
        ["Tema 1 — Comportamiento de clientes y fidelización",
         "Tema 2 — Preferencias de pago y su impacto"]
    )

    # ---------- Tema 1 ----------
    if "Tema 1" in tema_pseudo:
        st.markdown("### 🧠 Tema 1 — Comportamiento de clientes y fidelización")
        st.code(
            """
INICIO
    CARGAR dataset de clientes
    CARGAR dataset de ventas
    UNIR ambos datasets POR id_cliente
    CALCULAR frecuencia_compra = cantidad de ventas por cliente
    CALCULAR recencia = fecha_actual - última_compra
    CALCULAR monetización = suma de importes por cliente
    CLASIFICAR clientes EN:
        - Nuevos (fecha_alta reciente)
        - Activos (recencia baja)
        - Inactivos (recencia alta)
    MOSTRAR métricas de fidelización
FIN
            """,
            language="text"
        )

    # ---------- Tema 2 ----------
    if "Tema 2" in tema_pseudo:
        st.markdown("### 💳 Tema 2 — Preferencias de pago y su impacto")
        st.code(
            """
INICIO
    CARGAR dataset de ventas
    AGRUPAR ventas POR medio_pago
    CONTAR cantidad de operaciones por método
    CALCULAR importe_total POR método de pago
    ORDENAR resultados de mayor a menor
    GENERAR gráfico de barras:
        - Eje X: medios de pago
        - Eje Y: cantidad de ventas o importes
    IDENTIFICAR el método más utilizado
    RECOMENDAR promociones basadas en los resultados
FIN
            """,
            language="text"
        )
        
# --------------------------------------
# DIAGRAMA
# --------------------------------------
if section == "Diagrama":
    st.subheader("🧭 Diagrama de flujo — App Tienda Aurelion")

    dot = r'''
    digraph G {
      bgcolor="transparent";
      rankdir=TB;  // 👈 orientación vertical (Top to Bottom)
      fontsize=10;

      node [shape=rectangle, style="rounded,filled", fillcolor="#F7F7F9",
            color="#B8B8C4", fontname="Helvetica", fontsize=10];
      edge [color="#73FF86", penwidth=1.8];  // 💚 color de las flechas

      start  [shape=circle, label="Inicio", fillcolor="#E8F5E9"];
      load   [label="Cargar datasets (BD/*.xlsx)\nload_excel_or_prompt()", fillcolor="#E3F2FD"];
      header [label="Header: LOGO (IMAGES/LOGO2.png)\n+ título/subtítulo", fillcolor="#F3E5F5"];
      nav    [label="Sidebar: radio('Temas','Fuentes','Pseudocódigo','Diagrama')", fillcolor="#FFFDE7"];

      temas   [label="Página: Temas", fillcolor="#E8EAF6"];
      tsel    [label="selectbox: Tema 1 / Tema 2", fillcolor="#E8EAF6"];
      t2      [label="Tema 1: KPIs + Top10 clientes\n(expander)", fillcolor="#E8EAF6"];
      t3      [label="Tema 2: Ventas por medio de pago\n(expander)", fillcolor="#E8EAF6"];

      fuentes [label="Página: Fuentes", fillcolor="#E0F2F1"];
      f1      [label="clientes.xlsx\n(schema_table + head)", fillcolor="#E0F2F1"];
      f2      [label="ventas.xlsx\n(schema_table + head)", fillcolor="#E0F2F1"];
      f3      [label="detalle_ventas.xlsx\n(schema_table + head)", fillcolor="#E0F2F1"];
      f4      [label="productos.xlsx\n(schema_table + head)", fillcolor="#E0F2F1"];

      pseudo  [label="Página: Pseudocódigo", fillcolor="#FFF3E0"];
      psel    [label="selectbox: Tema 1 / Tema 2", fillcolor="#FFF3E0"];
      pc2     [label="Pseudocódigo Tema 1", fillcolor="#FFF3E0"];
      pc3     [label="Pseudocódigo Tema 2", fillcolor="#FFF3E0"];

      start -> load -> header -> nav;

      nav -> temas;
      nav -> fuentes;
      nav -> pseudo;
      nav -> start [style=dotted, color="#F291FF",fontcolor="#F291FF", label="(volver)", fontsize=10];

      temas -> tsel;
      tsel -> t2;
      tsel -> t3;

      fuentes -> f1;
      fuentes -> f2;
      fuentes -> f3;
      fuentes -> f4;

      pseudo -> psel;
      psel -> pc2;
      psel -> pc3;
    }
    '''

    st.graphviz_chart(dot, use_container_width=True)
    st.caption("El diagrama refleja el flujo actual de navegación y vistas de la app, sin análisis adicionales.")

# --------------------------------------
# RESUMEN SPRINT 1
# --------------------------------------
if section == "Resumen Sprint 1":
    st.subheader("📘 Resumen — Sprint 1")

    st.markdown(
        """
        ### 🛒 Tienda Aurelion — Aplicación Interactiva (Sprint 1)

        **Objetivo del Sprint:**  
        Construir la base funcional de la aplicación interactiva en Streamlit, permitiendo visualizar datos,
        navegar entre secciones y presentar los fundamentos analíticos del proyecto *Tienda Aurelion*.

        ---

        ### ✅ Entregables logrados

        - Estructura completa del proyecto **AURELION/**
          - Subcarpetas organizadas: `BD/` y `IMAGES/`
          - Archivo principal: `aurelion_app.py`
          - Documentación: `requirements.txt` y `README.md`
        - **Interfaz Streamlit** con encabezado, logo y navegación lateral.
        - **Carga dinámica** de datasets Excel (`clientes`, `ventas`, `productos`, `detalle_ventas`).
        - **Visualización de temas del TP**:
          - *Tema 1:* Comportamiento de clientes y fidelización.  
          - *Tema 2:* Preferencias de pago y su impacto.
        - **Pseudocódigo y Diagrama de flujo** integrados en la app.
        - **Esquema de datos** automático con tipo y escala estimada.

        ---

        ### 💡 Próximos pasos — Sprint 2 (Análisis)

        - Incorporar **indicadores RFM (Recencia, Frecuencia, Monetización)**.
        - Agregar **gráficos interactivos** (barras, líneas, tortas, mapas de calor).
        - Crear paneles de **insights automáticos** y **segmentación de clientes**.
        - Integrar **descargas CSV** y comparativas entre períodos.
        - Mejorar la **presentación visual** (temas, colores, disposición).

        ---

        ### 👤 Autor

        **Leandro Serantes**   
        """
    )
# ================================
# SPRINT 2 — Sección: Análisis de datos
# ================================
if section == "Análisis de datos":
    st.subheader("SPRINT 2 — Análisis de datos")
    st.caption("Revisión por dataset: valores faltantes, duplicados, inconsistencias, atípicos y tipos incorrectos.")

    

    # ---------- Helpers ----------
    def normalize_categorical(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_object_dtype(s):
            return s
        # normaliza espacios, mayúsc/minúsc, separadores comunes
        return (
            s.astype(str)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
             .str.replace(".", "/", regex=False)
             .str.replace("-", "/", regex=False)
             .str.lower()
        )

    def check_missing(df: pd.DataFrame) -> pd.DataFrame:
        miss = df.isna().sum().reset_index()
        miss.columns = ["columna", "faltantes"]
        miss["%_faltantes"] = (miss["faltantes"] / len(df) * 100).round(2)
        return miss.sort_values("%_faltantes", ascending=False)

    def check_duplicates(df: pd.DataFrame, sample_n: int = 10):
        dup_mask = df.duplicated(keep=False)
        n_dups = int(dup_mask.sum())
        sample = df.loc[dup_mask].head(sample_n)
        return n_dups, sample

    def check_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                raw_unique = df[col].dropna().unique()
                norm = normalize_categorical(df[col].dropna())
                norm_unique = norm.unique()
                # si normalizar reduce únicos, hay inconsistencias de formato/capitalización/espacios
                if len(norm_unique) < len(raw_unique):
                    rows.append({
                        "columna": col,
                        "unicos_raw": len(raw_unique),
                        "unicos_normalizados": len(norm_unique),
                        "reduccion": len(raw_unique) - len(norm_unique)
                    })
        return pd.DataFrame(rows).sort_values("reduccion", ascending=False) if rows else pd.DataFrame(
            columns=["columna", "unicos_raw", "unicos_normalizados", "reduccion"]
        )

    def check_wrong_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta columnas objeto que podrían ser numéricas o fechas.
        """
        out = []
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_object_dtype(s):
                # ¿se puede convertir a número?
                num_try = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
                pct_num = (num_try.notna().mean() * 100).round(2)

                # ¿se puede convertir a fecha?
                dt_try = pd.to_datetime(
                    s.str.replace(".", "/", regex=False).str.replace("-", "/", regex=False),
                    errors="coerce",
                    dayfirst=True
                )
                pct_dt = (dt_try.notna().mean() * 100).round(2)

                if pct_num >= 50 or pct_dt >= 50:  # umbral configurable
                    out.append({
                        "columna": col,
                        "dtype_actual": str(s.dtype),
                        "convertible_a_num_%": pct_num,
                        "convertible_a_fecha_%": pct_dt
                    })
        return pd.DataFrame(out).sort_values(["convertible_a_num_%", "convertible_a_fecha_%"], ascending=False) if out else pd.DataFrame(
            columns=["columna", "dtype_actual", "convertible_a_num_%", "convertible_a_fecha_%"]
        )
    #HACIENDO EL CAMBBIO
    def check_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Detecta outliers por método IQR en columnas numéricas.
        Devuelve:
        - Un DataFrame resumen con conteos y porcentajes
        - Un diccionario {columna: DataFrame de filas outlier}
        """
        

        num_cols = df.select_dtypes(include=[np.number]).columns
        resumen_rows = []
        outlier_samples = {}

        for col in num_cols:
            x = df[col].dropna()
            if x.empty:
                continue
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            mask = (df[col] < low) | (df[col] > high)
            cnt = int(mask.sum())

            if cnt > 0:
                resumen_rows.append({
                    "columna": col,
                    "outliers": cnt,
                    "%_outliers_sobre_no_na": round(cnt / len(x) * 100, 2),
                    "min": round(x.min(), 2),
                    "q1": round(q1, 2),
                    "q3": round(q3, 2),
                    "max": round(x.max(), 2)
                })
                # Guardamos los registros outlier para mostrar luego
                outlier_samples[col] = df.loc[mask, [col]].copy()

        resumen_df = pd.DataFrame(resumen_rows).sort_values(
            "%_outliers_sobre_no_na", ascending=False
        ) if resumen_rows else pd.DataFrame(
            columns=["columna", "outliers", "%_outliers_sobre_no_na", "min", "q1", "q3", "max"]
        )

        return resumen_df, outlier_samples

    # ---------- Datasets a revisar ----------
    tablas = {
        "clientes": clientes,
        "ventas": ventas,
        "detalle_ventas": detalle_ventas,
        "productos": productos
    }

    resumen_global = []

    for nombre, df in tablas.items():
        st.markdown(f"### 📁 {nombre}")
        colA, colB, colC, colD = st.columns(4)
        n_rows, n_cols = df.shape
        total_missing = int(df.isna().sum().sum())
        n_dups, _sample_dups = check_duplicates(df)
        # outliers totales
        out_df, out_samples = check_outliers(df)
        total_outliers = int(out_df["outliers"].sum()) if not out_df.empty else 0

        colA.metric("Filas", n_rows)
        colB.metric("Columnas", n_cols)
        colC.metric("Celdas faltantes", total_missing)
        colD.metric("Filas duplicadas", n_dups)

        with st.expander("🔎 Valores faltantes (por columna)", expanded=False):
            st.dataframe(check_missing(df), use_container_width=True, hide_index=True)

        with st.expander("📄 Duplicados exactos (muestra)", expanded=False):
            st.write(f"**Total de filas duplicadas (incluye todos los duplicados):** {n_dups}")
            if n_dups > 0:
                st.dataframe(_sample_dups, use_container_width=True)

        with st.expander("⚠️ Inconsistencias de formato en categóricas", expanded=False):
            inconsist = check_inconsistencies(df)
            if inconsist.empty:
                st.success("No se detectaron inconsistencias relevantes en columnas categóricas.")
            else:
                st.dataframe(inconsist, use_container_width=True, hide_index=True)
                st.caption("Sugerencia: normalizar con `.str.strip().str.lower()` y unificar separadores.")

        with st.expander("🧪 Tipos potencialmente incorrectos (texto que parece número/fecha)", expanded=False):
            wrong = check_wrong_types(df)
            if wrong.empty:
                st.success("No se detectaron columnas de texto con alta convertibilidad a número/fecha.")
            else:
                st.dataframe(wrong, use_container_width=True, hide_index=True)
                st.caption("Sugerencia: convertir a numérico con `pd.to_numeric(..., errors='coerce')` o a fecha con `pd.to_datetime(..., dayfirst=True)`.")

        with st.expander("📉 Valores atípicos (IQR) en numéricas", expanded=False):
            if out_df.empty:
                st.info("No hay columnas numéricas o no se detectaron outliers por IQR.")
            else:
                st.dataframe(out_df, use_container_width=True, hide_index=True)
                
                if not out_df.empty and out_samples:
                    colnames = list(out_samples.keys())
                    if colnames:  # por si acaso
                        sel_col = st.selectbox("📍 Ver registros outlier de:", colnames, key=f"outsel_{nombre}")
                        st.dataframe(out_samples[sel_col], use_container_width=True)

                # 🎨 Solo para detalle_ventas: graficar boxplot del campo "importe"
                if nombre == "detalle_ventas" and "importe" in df.columns:
                    

                    x = df["importe"].dropna()

                    # Calcular Q1, Q3, IQR y límites
                    q1 = x.quantile(0.25)
                    q3 = x.quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - 1.5 * iqr
                    high = q3 + 1.5 * iqr      

                    fig, ax = plt.subplots(figsize=(7, 0.8))
                    ax.boxplot(
                        x,
                        vert=False,
                        patch_artist=True,
                        boxprops=dict(facecolor="#00FF22", edgecolor="white"),
                        medianprops=dict(color="white", linewidth=1.2),
                        whiskerprops=dict(color="white", linewidth=1),
                        capprops=dict(color="white", linewidth=1),
                        flierprops=dict(marker='o', markerfacecolor='white', markersize=3, markeredgecolor='white')
                    )
                    ax.tick_params(axis="x", colors="white", labelsize=8)
                    ax.tick_params(axis="y", colors="white", labelsize=8)
                    ax.spines["bottom"].set_color("white")
                    ax.spines["top"].set_color("white")
                    ax.spines["right"].set_color("white")
                    ax.spines["left"].set_color("white")
                    ax.set_title("Distribución de importes — Detalle de Ventas", fontsize=8, color="white")
                    ax.set_xlabel("Importe ($)",fontsize=8, color="white")
                    ax.axvline(low, color="red", linestyle="--", linewidth=1)
                    ax.axvline(high, color="red", linestyle="--", linewidth=1)
                    st.pyplot(fig, transparent=True)

                    st.caption(
                        f"Boxplot de importes. Los valores por encima de **${high:,.0f}** "
                        f"se consideran outliers (encontrados: {int((x > high).sum())})."
                    )



        resumen_global.append({
            "tabla": nombre,
            "filas": n_rows,
            "columnas": n_cols,
            "celdas_faltantes": total_missing,
            "filas_duplicadas": n_dups,
            "outliers_detectados": total_outliers,
        })

        st.markdown("---")

    st.markdown("## 🧾 Resumen de hallazgos")
    st.dataframe(pd.DataFrame(resumen_global), use_container_width=True, hide_index=True)


# ================================
# SPRINT 2 — Sección: Modelo de Datos
# ================================
if section == "Modelo de Datos":
    st.subheader("SPRINT 2 — Modelo de Datos")
    st.caption("Diagrama relacional con claves primarias (🔑) y foráneas (🔗).")

    dot = r'''
    digraph G {
    bgcolor="transparent";
    rankdir=LR;
    graph [pad="0.2"];
    node  [shape=plaintext, fontname="Helvetica", fontcolor="white"];
    edge  [color="#73FF86", penwidth=2, arrowsize=0.8, fontcolor="white"];

    clientes [
        label=<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="white">
            <TR><TD BGCOLOR="#4B6EAF" COLOR="white"><B><FONT COLOR="white">clientes</FONT></B></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">🔑 id_cliente</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">nombre_cliente</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">email</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">ciudad</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">fecha_alta</FONT></TD></TR>
        </TABLE>
        >
    ];

    ventas [
        label=<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="white">
            <TR><TD BGCOLOR="#AA8539"><B><FONT COLOR="white">ventas</FONT></B></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">🔑 id_venta</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">fecha</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">🔗 id_cliente</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">nombre_cliente</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">email</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">medio_pago</FONT></TD></TR>
        </TABLE>
        >
    ];

    detalle_ventas [
        label=<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="white">
            <TR><TD BGCOLOR="#9C8E37"><B><FONT COLOR="white">detalle_ventas</FONT></B></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">🔗 id_venta</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">🔗 id_producto</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">nombre_producto</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">cantidad</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">precio_unitario</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">importe</FONT></TD></TR>
        </TABLE>
        >
    ];

    productos [
        label=<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="white">
            <TR><TD BGCOLOR="#4B6EAF"><B><FONT COLOR="white">productos</FONT></B></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">🔑 id_producto</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">nombre_producto</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">categoria</FONT></TD></TR>
            <TR><TD ALIGN="LEFT"><FONT COLOR="white">precio_unitario</FONT></TD></TR>
        </TABLE>
        >
    ];

    // Relaciones (FK)
    clientes -> ventas        [label=" id_cliente "];
    ventas   -> detalle_ventas[label=" id_venta "];
    detalle_ventas-> productos[label=" id_producto "];
    }
    '''



    st.graphviz_chart(dot, use_container_width=True)



# ================================
# SPRINT 2 — Sección: Distribución de Variables
# ================================
if section == "Distribución de Variables":
    st.subheader("SPRINT 2 — Identificación del tipo de distribución de variables")
    st.caption("Análisis visual y estadístico de las variables numéricas principales (importe, cantidad, precio_unitario).")

    # --- Selección de tabla y columna ---
    tablas_num = {
        "Ventas": ventas.select_dtypes(include=["number"]),
        "Detalle de Ventas": detalle_ventas.select_dtypes(include=["number"]),
        "Productos": productos.select_dtypes(include=["number"])
    }

    tabla_sel = st.selectbox("📁 Elegí la tabla a analizar:", list(tablas_num.keys()))
    df_sel = tablas_num[tabla_sel]

    if df_sel.empty:
        st.warning("La tabla seleccionada no contiene variables numéricas para analizar.")
    else:
        col_sel = st.selectbox("📊 Elegí la variable numérica:", df_sel.columns)

        # --- Gráfico + interpretación lateral con explicaciones detalladas ---
        
        
        
        from scipy.stats import skew, kurtosis

        x = df_sel[col_sel].dropna()
        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        sns.histplot(x, kde=True, color="#73FF86", bins=20, ax=ax)
        ax.set_title(f"Distribución de {col_sel}", fontsize=9, color="white")
        ax.set_xlabel(col_sel, fontsize=8, color="white")
        ax.tick_params(colors="white", labelsize=8)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        # 📊 Estadísticos principales
        media = x.mean()
        mediana = x.median()
        desv = x.std()
        asim = skew(x)
        curt = kurtosis(x)
        
        

        # --- Interpretaciones automáticas por métrica ---
        exp_media   = "Promedio general de los valores. Un valor alto indica que la mayoría de observaciones son grandes."
        exp_mediana = "Valor central. Si difiere mucho de la media, sugiere una distribución sesgada."
        exp_desv    = ("Mide la dispersión. Cuanto mayor es, más alejados están los valores del promedio."
                        if desv > 0 else "No hay variación entre los valores (todos son iguales).")

        if asim > 1:
            exp_asim = "Sesgo a la derecha → predominan valores bajos y algunos muy altos."
        elif asim < -1:
            exp_asim = "Sesgo a la izquierda → predominan valores altos y algunos muy bajos."
        else:
            exp_asim = "Distribución aproximadamente simétrica (sin sesgo marcado)."

        if curt > 3:
            exp_curt = "Colas pesadas → existen valores extremos o atípicos significativos."
        elif curt < 3:
            exp_curt = "Colas planas → la mayoría de los valores están concentrados cerca del promedio."
        else:
            exp_curt = "Curtosis normal (mesocúrtica) → distribución balanceada."

        # --- Interpretación general combinada (si la querés usar en otro lado) ---
        interpretacion = ""
        if asim > 1:
            interpretacion += "Distribución sesgada a la **derecha** (valores altos más dispersos).  \n"
        elif asim < -1:
            interpretacion += "Distribución sesgada a la **izquierda** (valores bajos más dispersos).  \n"
        else:
            interpretacion += "Distribución **aproximadamente simétrica**.  \n"

        if curt > 3:
            interpretacion += "Colas **más pesadas** que la normal (mayor presencia de valores extremos)."
        elif curt < 3:
            interpretacion += "Colas **más planas** que la normal (más uniforme)."
        else:
            interpretacion += "Curtosis similar a la **normal (mesocúrtica)**."

        # ---------- Helpers para la interpretación específica ----------
        def _nivel_dispersion(std, mean):
            if mean == 0 or np.isnan(std) or np.isnan(mean):
                return "baja"
            ratio = std / mean
            if ratio < 0.25: return "baja"
            if ratio < 0.60: return "moderada"
            return "alta"

        def _sesgo_text(a):
            if a > 1:     return "Sesgo <b>fuerte a la derecha</b>: pocas observaciones muy altas elevan el promedio."
            if a > 0.2:   return "Sesgo <b>leve a la derecha</b>: hay algunas compras/montos altos."
            if a < -1:    return "Sesgo <b>fuerte a la izquierda</b>: predominan valores altos con pocos muy bajos."
            if a < -0.2:  return "Sesgo <b>leve a la izquierda</b>."
            return "Distribución <b>aproximadamente simétrica</b>."

        def _curt_text(c):
            if c > 3:   return "Colas <b>pesadas</b> (más outliers que lo normal)."
            if c < 3:   return "Colas <b>planas</b> (más uniforme, menos extremos)."
            return "Curtosis <b>normal (mesocúrtica)</b>."

        def interpretar_variable(nombre_col, serie, media, mediana, desv, asim, curt):
            n = str(nombre_col).lower()
            disp     = _nivel_dispersion(desv, media)
            sesgo    = _sesgo_text(asim)
            curtosis = _curt_text(curt)
            minimo, maximo = float(np.nanmin(serie)), float(np.nanmax(serie))

            # Identificadores
            if n.startswith("id_"):
                return f"""
                <p><b>Qué mide:</b> identificador técnico (<code>{nombre_col}</code>).</p>
                <p><b>Lectura:</b> se espera distribución <b>uniforme / casi uniforme</b>.
                Sirve para integridad de datos; no aporta insight de negocio directo.</p>
                <p><b>Rango observado:</b> {int(minimo)} – {int(maximo)}.</p>
                """

            # Importe
            if "importe" in n:
                punta = ("mayoría en valores <b>bajos/medios</b> con algunas ventas grandes"
                         if (mediana < media and asim > 0)
                         else "reparto relativamente equilibrado"
                         if abs(media - mediana) / (media + 1e-9) < 0.1
                         else "mayoría en valores <b>medios/altos</b>")
                return f"""
                <p><b>Qué mide:</b> monto por línea de venta.</p>
                <p><b>Dispersión:</b> {disp}. {sesgo} {curtosis}</p>
                <p><b>Lectura de negocio:</b> {punta}. Útil para detectar
                <b>clientes/ventas premium</b> y definir <b>umbrales de outliers</b>.</p>
                <p><b>Rango observado:</b> ${minimo:,.0f} – ${maximo:,.0f}.</p>
                """

            # Cantidad
            if "cantidad" in n:
                low_bulk = "compras unitarias dominan" if media <= 2 else "mezcla de compras chicas y medianas"
                return f"""
                <p><b>Qué mide:</b> unidades por ítem/venta.</p>
                <p><b>Dispersión:</b> {disp}. {sesgo} {curtosis}</p>
                <p><b>Lectura de negocio:</b> {low_bulk}.
                Sirve para definir <b>packs/promos por volumen</b> y detectar <b>compras atípicas</b>.</p>
                <p><b>Rango observado:</b> {minimo:,.0f} – {maximo:,.0f}.</p>
                """

            # Precio unitario
            if "precio_unit" in n:
                premium = "existen productos <b>premium</b> (cola derecha)" if asim > 0.2 else "estructura de precios compacta"
                return f"""
                <p><b>Qué mide:</b> precio del producto.</p>
                <p><b>Dispersión:</b> {disp}. {sesgo} {curtosis}</p>
                <p><b>Lectura de negocio:</b> {premium}.
                Útil para <b>segmentar por categoría/precio</b> y evaluar <b>mix de productos</b>.</p>
                <p><b>Rango observado:</b> ${minimo:,.0f} – ${maximo:,.0f}.</p>
                """

            # Genérico
            return f"""
            <p><b>Dispersión:</b> {disp}. {sesgo} {curtosis}</p>
            <p><b>Lectura:</b> revisar outliers si hay colas pesadas; comparar media vs mediana para detectar sesgos.</p>
            <p><b>Rango observado:</b> {minimo:,.2f} – {maximo:,.2f}.</p>
            """

        # --- Layout visual (dos columnas) ---
        col1, col2 = st.columns([2, 1.3])

        with col1:
            st.pyplot(fig, transparent=True)

        with col2:
            st.markdown("### 🧭 Resumen interpretativo")
            st.markdown(f"**Variable analizada:** `{col_sel}`")

            st.markdown(f"- **Media:** {media:,.2f}")
            st.caption(f"📘 {exp_media}")

            st.markdown(f"- **Mediana:** {mediana:,.2f}")
            st.caption(f"📘 {exp_mediana}")

            st.markdown(f"- **Desviación estándar:** {desv:,.2f}")
            st.caption(f"📘 {exp_desv}")

            st.markdown(f"- **Asimetría (Skewness):** {asim:.2f}")
            st.caption(f"📘 {exp_asim}")

            st.markdown(f"- **Curtosis (Kurtosis):** {curt:.2f}")
            st.caption(f"📘 {exp_curt}")

            # 🔽 Caja de interpretación específica (HTML, sin Markdown dentro)
            interpretacion_html = interpretar_variable(col_sel, x.values, media, mediana, desv, asim, curt)
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 10px;
                    padding: 14px 16px;
                    margin-top: 10px;
                    color: white;
                    font-size: 14px;
                    line-height: 1.6;
                ">
                    <h4 style="color:#73FF86; margin-bottom:8px;">🧾 Interpretación específica</h4>
                    {interpretacion_html}
                </div>
                """,
                unsafe_allow_html=True
            )        
            
        

        


# ================================
# SPRINT 2 — Sección: Correlaciones entre variables principales
# ================================
if section == "Correlaciones entre variables principales":
    st.subheader("SPRINT 2 — Correlaciones entre variables principales")
    st.caption("Matriz de correlación y visualización de relaciones entre variables numéricas.")

    
    
    
    

    tablas_corr = {
        "Ventas": ventas,
        "Detalle de Ventas": detalle_ventas,
        "Productos": productos
    }

    tcol1, tcol2 = st.columns([1.1, 1])
    with tcol1:
        tabla_sel = st.selectbox("📁 Elegí la tabla:", list(tablas_corr.keys()), key="corr_tabla")
    df_base = tablas_corr[tabla_sel]
    df_num = df_base.select_dtypes(include=[np.number]).copy()

    if df_num.shape[1] < 2:
        st.warning("La tabla seleccionada no tiene 2 o más variables numéricas para correlacionar.")
    else:
        colnames = list(df_num.columns)

        c1, c2 = st.columns([1.3, 1])
        with c1:
            vars_sel = st.multiselect("🔢 Variables a incluir (dejar vacío para usar todas):", colnames, default=colnames)
        with c2:
            metodo = st.radio("Método", ["Pearson", "Spearman"], horizontal=True, key="corr_metodo")

        if not vars_sel:
            vars_sel = colnames

        corr = df_num[vars_sel].corr(method=metodo.lower())

        st.markdown("### 📋 Matriz de correlación")
        st.dataframe(corr.round(2), use_container_width=True)

        # --- Heatmap (modo oscuro + fondo transparente) ---
        # --- Heatmap (modo oscuro + fondo transparente, versión compacta tipo “miniatura”) ---
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)  # tamaño más chico pero nítido
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap="vlag",
            annot=True, fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.4, linecolor="gray",
            cbar_kws={"shrink": 0.6},
            ax=ax
        )
        # 🔧 Ajustar color de texto en la barra de color
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white", fontsize=7)
        cbar.set_label("Coeficiente de correlación", color="white", fontsize=8, labelpad=6)



        ax.set_title(
            f"Matriz de correlación — {tabla_sel} ({metodo})",
            color="white", fontsize=9, pad=6
        )
        ax.tick_params(colors="white", labelsize=7)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("white")

        st.pyplot(fig, transparent=True, use_container_width=False)

        # --- Interpretación automática ---
        # Buscamos los pares más correlacionados (positivos y negativos)
        pairs = (
            corr.where(~np.eye(corr.shape[0], dtype=bool))
                .stack()
                .reset_index()
        )
        pairs.columns = ["var1", "var2", "corr"]
        pairs["abs_corr"] = pairs["corr"].abs()
        top_pair = pairs.loc[pairs["abs_corr"].idxmax()]

        # Texto interpretativo
        if abs(top_pair["corr"]) < 0.3:
            rel = "muy baja o inexistente"
        elif abs(top_pair["corr"]) < 0.5:
            rel = "moderada"
        elif abs(top_pair["corr"]) < 0.8:
            rel = "fuerte"
        else:
            rel = "muy fuerte"

        sentido = "positiva (directa)" if top_pair["corr"] > 0 else "negativa (inversa)"

        st.info(
            f"📊 **Interpretación automática:** Las variables **{top_pair['var1']}** y **{top_pair['var2']}** "
            f"presentan una correlación **{rel}** y **{sentido}** "
            f"(coeficiente = {top_pair['corr']:.2f})."
        )


        # --- Top pares correlacionados ---
        st.markdown("### 🔎 Pares más correlacionados")
        thr = st.slider("Umbral de |correlación|", 0.0, 1.0, 0.5, 0.05, help="Mostrar pares con correlación absoluta mayor o igual al umbral.")
        # apilar, eliminar diagonal y duplicados (A-B y B-A)
        pairs = (
            corr.where(~np.eye(corr.shape[0], dtype=bool))
                .stack()
                .reset_index()
        )
        pairs.columns = ["var1", "var2", "corr"]
        # evitar duplicados simétricos
        pairs["key"] = pairs.apply(lambda r: tuple(sorted([r["var1"], r["var2"]])), axis=1)
        pairs = pairs.drop_duplicates("key").drop(columns="key")
        pairs["abs"] = pairs["corr"].abs()
        top_pairs = pairs[pairs["abs"] >= thr].sort_values("abs", ascending=False)

        if top_pairs.empty:
            st.info("No hay pares que superen el umbral seleccionado.")
        else:
            st.dataframe(top_pairs[["var1", "var2", "corr"]].reset_index(drop=True).round(2), use_container_width=True)

            # --- Dispersión para un par elegido ---
            st.markdown("#### 📈 Dispersión del par seleccionado")
            par = st.selectbox(
                "Elegí un par para visualizar la relación (scatter):",
                [f"{r.var1} ⇄ {r.var2}" for _, r in top_pairs.iterrows()],
                key="par_scatter"
            )
            v1, v2 = [s.strip() for s in par.split("⇄")]
            x = df_num[v1]
            y = df_num[v2]

            # --- Dispersión para un par elegido (versión compacta y nítida) ---
            fig2, ax2 = plt.subplots(figsize=(4, 2.6), dpi=140)  # 👈 más chico + alta resolución
            fig2.patch.set_alpha(0.0)
            ax2.set_facecolor("none")

            ax2.scatter(
                x, y,
                alpha=0.7,
                s=14,           # 👈 puntos un poco más chicos
                linewidths=0,
            )

            # Estética en modo oscuro
            for spine in ax2.spines.values():
                spine.set_color("white")
            ax2.tick_params(colors="white", labelsize=7)
            ax2.set_xlabel(v1, color="white", fontsize=8)
            ax2.set_ylabel(v2, color="white", fontsize=8)
            ax2.set_title(f"Relación {v1} vs {v2}", color="white", fontsize=9, pad=4)

            # Render sin expandir (no ocupa todo el ancho)
            st.pyplot(fig2, transparent=True, use_container_width=False)


        st.caption("Nota: Pearson mide relación lineal; Spearman mide relación monótona (usa rangos).")

# ================================
# SPRINT 2 — Sección: RFM (Segmentación clientes)
# ================================
if section == "RFM (Segmentación clientes)":
    st.subheader("SPRINT 2 — RFM (Recency, Frequency, Monetary)")
    st.caption("Segmentación por recencia de compra, frecuencia de compras y gasto monetario total por cliente.")

    
    
    
    

    # --- Configuración ---
    # Fecha de referencia: última fecha disponible en ventas (fallback: hoy)
    fecha_ref = pd.to_datetime(ventas["fecha"], dayfirst=True, errors="coerce").max()
    if pd.isna(fecha_ref):
        fecha_ref = pd.Timestamp.today().normalize()

    # Unimos detalle con ventas para tener cliente y fechas por cada línea
    vcols = ["id_venta", "id_cliente", "fecha"]
    _ventas_dt = ventas.copy()
    _ventas_dt["fecha"] = pd.to_datetime(_ventas_dt["fecha"], dayfirst=True, errors="coerce")

    dv_join = (
        detalle_ventas
        .merge(_ventas_dt[vcols], on="id_venta", how="left", validate="m:1")
        .dropna(subset=["id_cliente"])
    )

    # --- Agregaciones por cliente ---
    # Frequency = # de órdenes únicas
    freq = dv_join.groupby("id_cliente")["id_venta"].nunique().rename("frequency")

    # Monetary = suma de importe (si no existe importe, calculamos cantidad*precio_unitario)
    if "importe" in dv_join.columns and np.issubdtype(dv_join["importe"].dtype, np.number):
        mon = dv_join.groupby("id_cliente")["importe"].sum().rename("monetary")
    else:
        mon = (dv_join["cantidad"] * dv_join["precio_unitario"]).groupby(dv_join["id_cliente"]).sum().rename("monetary")

    # Recency = días desde última compra
    last_date = dv_join.groupby("id_cliente")["fecha"].max().rename("last_purchase_date")
    rec = (fecha_ref - last_date).dt.days.rename("recency_days")

    rfm = pd.concat([rec, freq, mon], axis=1).fillna({"recency_days": np.inf, "frequency": 0, "monetary": 0})
    rfm = rfm.reset_index().rename(columns={"id_cliente": "cliente"})

    # --- Scoring por cuantiles (config fijo) ---
    BINS_RFM = 5  # Elegí 3, 4 o 5. Recomendado: 5.


    

    

    def score_qcut(s: pd.Series, q: int):
        """
        Scoring tipo cuantiles, robusto a empates.
        1 = bajo/peor ... q = alto/mejor (si querés invertir, hacelo antes con .rank(ascending=False)).
        """
        x = pd.to_numeric(s, errors="coerce")
        # percentil-rank en [0,1]
        r = x.rank(method="average", pct=True)
        # bordes uniformes fijos → evita 'duplicates=drop' y el ValueError
        edges = np.linspace(0.0, 1.0, q + 1)
        labels = list(range(1, q + 1))
        return pd.cut(r, bins=edges, labels=labels, include_lowest=True)


    # Recency: menos días es mejor ⇒ invertimos
    r_scores = score_qcut(rfm["recency_days"].rank(ascending=False), BINS_RFM).astype(int)
    f_scores = score_qcut(rfm["frequency"], BINS_RFM).astype(int)
    m_scores = score_qcut(rfm["monetary"],  BINS_RFM).astype(int)


    rfm["R"] = r_scores
    rfm["F"] = f_scores
    rfm["M"] = m_scores
    rfm["RFM_score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["RFM_sum"] = rfm[["R","F","M"]].sum(axis=1)

    def segment_row(r, q=BINS_RFM):
        high = q
        if (r.R >= high-0) and (r.F >= high-1) and (r.M >= high-1):
            return "Champions"
        if (r.R >= high-1) and (r.F >= high-1):
            return "Leales"
        if (r.R >= high-1) and (r.M >= high-1):
            return "Grandes Gastadores"
        if (r.R <= 2) and (r.F <= 2) and (r.M <= 2):
            return "En Riesgo / Dormidos"
        return "Potenciales"

    # 👇 garantizar que exista la columna
    if "segmento" not in rfm.columns:
        rfm["segmento"] = rfm.apply(segment_row, axis=1)


    # Join opcional con clientes para mostrar nombre/email si existen
    if "id_cliente" in clientes.columns:
        cols_disp = ["cliente","R","F","M","RFM_sum","RFM_score","segmento","recency_days","frequency","monetary"]
        rfm_display = (
            rfm.merge(clientes[["id_cliente","nombre_cliente","email"]],
                    left_on="cliente", right_on="id_cliente", how="left")
            .drop(columns=["id_cliente"])
            .rename(columns={"nombre_cliente":"cliente_nombre"})
        )
        desired = ["cliente","cliente_nombre","email"] + cols_disp[1:]  # respeta tu orden
        existing = [c for c in desired if c in rfm_display.columns]     # 👈 evita KeyError
        rfm_display = rfm_display[existing]
    else:
        rfm_display = rfm.copy()
        
    # =======================
    # Actividad de clientes (histórica)
    # =======================
    st.markdown("### 🧭 Actividad de clientes (histórica)")

    # clientes con al menos 1 venta registrada
    activos_ids = set(ventas["id_cliente"].dropna().unique())
    # universo de clientes (si no existiera 'clientes', usamos los del join)
    if "id_cliente" in clientes.columns:
        universo_ids = set(clientes["id_cliente"].dropna().unique())
    else:
        universo_ids = set(dv_join["id_cliente"].dropna().unique())

    inactivos_ids = universo_ids - activos_ids

    n_activos = len(activos_ids & universo_ids)
    n_inactivos = len(inactivos_ids)
    total_clientes = n_activos + n_inactivos

    colA, colB, colC = st.columns(3)
    colA.metric("Clientes (total)", total_clientes)
    colB.metric("Con actividad", n_activos, f"{(n_activos/total_clientes*100):.1f}%")
    colC.metric("Sin actividad", n_inactivos, f"{(n_inactivos/total_clientes*100):.1f}%")

    # Gráfico de torta
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # 🔹 gráfico más chico
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    labels = ["Con actividad", "Sin actividad"]
    sizes = [n_activos, n_inactivos]
    colors = ["#00640D", "#6B7280"]  # verde y gris oscuro

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.8,
        radius=0.7,  # 🔹 reduce el radio del círculo
        textprops={"color": "white", "fontsize": 8},
        wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )

    for t in autotexts:
        t.set_color("white")
    ax.axis("equal")

    # Mostrar gráfico centrado y reducido
    with st.container():
        st.markdown(
            """
            <div style='display:flex; justify-content:center;'>
                <div style='width:50%; min-width:250px;'>
            """,
            unsafe_allow_html=True
        )

        st.pyplot(fig, transparent=True, use_container_width=False)

        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.caption(
        "Definición: *Con actividad* = clientes con al menos una venta registrada en el histórico. "
        "*Sin actividad* = clientes presentes en el padrón sin ventas asociadas."
    )
        


    st.markdown("### 📋 Tabla RFM por cliente")
    st.dataframe(rfm_display.sort_values("RFM_sum", ascending=False), use_container_width=True)
    
    # ---- Botón "MÁS INFORMACIÓN" con toggle ----
    if "rfm_more" not in st.session_state:
        st.session_state.rfm_more = False

    label_more = "➖ OCULTAR INFORMACIÓN" if st.session_state.rfm_more else "➕ MÁS INFORMACIÓN"
    if st.button(label_more, key="btn_rfm_more"):
        st.session_state.rfm_more = not st.session_state.rfm_more

    if st.session_state.rfm_more:
        st.markdown(
            """
    | Segmento                    | Condición                                                             | Significado                                                | Estrategia típica                                       |
    | --------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
    | **🏆 Champions**            | `R` muy alto y `F` y `M` también altos                                | Compran seguido, gastan mucho, y lo hicieron recientemente | Recompensar su fidelidad, ofrecer beneficios exclusivos |
    | **💎 Leales**               | `R` y `F` altos                                                       | Compran con frecuencia, aunque no siempre con montos altos | Mantenerlos activos con ofertas o membresías            |
    | **💰 Grandes Gastadores**   | `R` y `M` altos                                                       | Gastaron mucho recientemente, aunque no tan seguido        | Promover productos premium o complementarios            |
    | **🌙 En Riesgo / Dormidos** | `R`, `F` y `M` bajos (poca frecuencia, poco gasto, hace mucho tiempo) | Clientes que se están alejando                             | Reactivar con descuentos o campañas personalizadas      |
    | **✨ Potenciales**           | Todo lo que no cae en las categorías anteriores                       | Clientes nuevos o irregulares                              | Incentivar segundas compras, newsletters o puntos       |
            """,
            help="Guía rápida de interpretación de segmentos RFM."
        )


    # --- Distribución por segmentos ---
    st.markdown("### 🧭 Distribución por segmentos")
    seg_counts = (
        rfm["segmento"]
        .value_counts()
        .rename_axis("segmento")
        .reset_index(name="clientes")
    )
    st.dataframe(seg_counts, use_container_width=True)

    # --- Gráfico: barras por segmento (compacto, centrado y con etiquetas) ---
    fig1, ax1 = plt.subplots(figsize=(5, 2.5))  # 🔹 más chico
    fig1.patch.set_alpha(0.0)
    ax1.set_facecolor("none")

    sns.barplot(data=seg_counts, x="segmento", y="clientes", ax=ax1, color="#189227")

    # Colores y estilo
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax1.tick_params(colors="white", labelrotation=20, labelsize=8)
    ax1.set_xlabel("Segmento", color="white", fontsize=9)
    ax1.set_ylabel("Clientes", color="white", fontsize=9)
    ax1.set_title("Clientes por segmento", color="white", fontsize=11)

    # --- Etiquetas sobre las barras ---
    for i, row in seg_counts.iterrows():
        ax1.text(
            i, 
            row["clientes"] + (row["clientes"] * 0.02),  # posición levemente arriba
            f"{int(row['clientes'])}", 
            ha="center", 
            color="white", 
            fontsize=7,
            fontweight="bold"
        )

    # ✅ Mostrar centrado con ancho reducido
    with st.container():
        st.markdown(
            """
            <div style='display:flex; justify-content:center;'>
                <div style='width:55%; min-width:280px;'>
            """,
            unsafe_allow_html=True
        )

        st.pyplot(fig1, transparent=True, use_container_width=False)

        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )



    # --- Gráfico: F vs M (color por R) ---
    st.markdown("### 📈 Frecuencia vs Monetario (color por Recencia)")

    # 🔹 Gráfico más compacto
    fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))  # antes (6, 3.5)
    fig2.patch.set_alpha(0.0)
    ax2.set_facecolor("none")

    sc = ax2.scatter(
        rfm["frequency"],
        rfm["monetary"],
        c=rfm["recency_days"],
        cmap="viridis_r",   # más claro = más reciente
        alpha=0.8,
        s=22
    )

    # Bordes invisibles para estilo limpio
    for spine in ax2.spines.values():
        spine.set_color("White")

    # Ejes y etiquetas
    ax2.tick_params(colors="white", labelsize=8)
    ax2.set_xlabel("Frecuencia (# órdenes)", color="white", fontsize=9)
    ax2.set_ylabel("Monetario ($)", color="white", fontsize=9)
    ax2.set_title("Relación Frecuencia–Monetario (más claro = más reciente)", color="white", fontsize=10)

    # --- Barra de color personalizada ---
    cbar = fig2.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white", fontsize=7)
    cbar.set_label(
        "← Más reciente   |   Más antiguo →",
        color="white",
        fontsize=8,
        labelpad=8
    )

    # ✅ Mostrar centrado y más pequeño
    with st.container():
        st.markdown(
            """
            <div style='display:flex; justify-content:center;'>
                <div style='width:60%; min-width:280px;'>
            """,
            unsafe_allow_html=True
        )

        st.pyplot(fig2, transparent=True, use_container_width=False)

        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        f"Fecha de referencia: {fecha_ref.date()}. "
        "Recency = días desde última compra (menos es mejor); "
        "Frequency = número de órdenes; Monetary = gasto total."
    )

    
    # ---- Botón "RESULTADOS" con resumen interpretativo ----
    if "rfm_results" not in st.session_state:
        st.session_state.rfm_results = False

    label_res = "📊 OCULTAR RESULTADOS" if st.session_state.rfm_results else "📈 MOSTRAR RESULTADOS"
    if st.button(label_res, key="btn_rfm_results"):
        st.session_state.rfm_results = not st.session_state.rfm_results

    if st.session_state.rfm_results:
        st.markdown(
            """
    ### 📊 Interpretación del gráfico Frecuencia–Monetario

    El gráfico muestra la relación entre la **frecuencia de compra (eje X)** y el **gasto total acumulado (eje Y)** de cada cliente, 
    con el **color indicando la recencia de la última compra**:

    - 🟡 **Tonos claros (amarillos)** → clientes **recientes**, que compraron hace pocos días.  
    - 🟣 **Tonos oscuros (violetas)** → clientes **antiguos**, que llevan mucho tiempo sin comprar.  
    - 📈 Los puntos **más altos y a la derecha** representan a los clientes más valiosos (frecuentes y de alto gasto).  
    - 🔍 Se observa que la mayoría de los clientes tienen **baja frecuencia (1 o 2 órdenes)**, pero dentro de ellos algunos alcanzan montos altos, mostrando **potenciales grandes compradores**.
    - 🧭 Los **clientes recientes y con gasto alto** son los **Champions**, mientras que los que aparecen abajo a la izquierda, de color más oscuro, están **en riesgo de abandono**.
            """
        )
        
    # ================================
    # 📘 CONCLUSIÓN Y PROPUESTA (Versión destacada)
    # ================================
    

# ================================
# SPRINT 2 — Sección: Resumen Sprint 2
# ================================
import textwrap as _tw

html_resumen_sprint2 = _tw.dedent("""
<div style="
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 20px 25px;
    margin-top: 10px;
    margin-bottom: 20px;
">
<h3 style="color: #73FF86; text-align: center;">📘 Resumen del Sprint 2 — Análisis de Datos</h3>

<p style="color: white; text-align: justify; font-size: 15px; line-height: 1.6;">
El <b>Sprint 2</b> tuvo como objetivo profundizar el análisis de la base de datos de la Tienda Aurelion mediante 
técnicas exploratorias y analíticas que permitieron evaluar la calidad, estructura y comportamiento de la información.  
A lo largo de este sprint se desarrollaron cinco secciones principales:
</p>

<p style="color: white; text-align: justify; font-size: 15px; line-height: 1.6;">
🧩 <b>1️⃣ Análisis de datos:</b> revisión de los datasets para detectar <b>valores faltantes, duplicados, inconsistencias, outliers</b> y <b>tipos incorrectos</b>.  
Se generaron métricas y visualizaciones que ayudaron a evaluar la integridad de la información.
<p style="color: white; text-align: justify; font-size: 15px; line-height: 1.6;">
🧭 <b>2️⃣ Modelo de Datos:</b> representación visual de las relaciones entre tablas mediante un <b>diagrama relacional</b> con claves primarias y foráneas, 
permitiendo comprender la estructura de la base.
</p>

<p style="color: white; text-align: justify; font-size: 15px; line-height: 1.6;">
📊 <b>3️⃣ Distribución de Variables:</b> identificación de los <b>tipos de distribución</b> y medidas de tendencia y dispersión 
(<b>media, mediana, desviación estándar, asimetría, curtosis</b>) con una interpretación automática para cada variable analizada.
</p>

<p style="color: white; text-align: justify; font-size: 15px; line-height: 1.6;">
🔗 <b>4️⃣ Correlaciones entre Variables:</b> análisis de las <b>relaciones lineales y monótonas</b> mediante matrices de correlación y gráficos de dispersión, 
incorporando una <b>interpretación automática</b> para detectar las variables más asociadas.
</p>

<p style="color: white; text-align: justify; font-size: 15px; line-height: 1.6;">
💎 <b>5️⃣ RFM — Segmentación de Clientes:</b> clasificación de clientes según <b>Recencia, Frecuencia y Valor Monetario</b>.  
Se identificaron segmentos estratégicos (<b>Champions, Leales, Grandes Gastadores, En Riesgo, Potenciales</b>) y se analizó el volumen de clientes inactivos, 
proponiendo estrategias de <b>fidelización y reactivación</b>.
</p>

<p style="color: white; text-align: justify; font-size: 15px; line-height: 1.6;">
✅ <b>Conclusión general:</b> el Sprint 2 permitió obtener una comprensión integral de la base de datos y sentó las bases para la creación 
de <b>indicadores estratégicos y dashboards interactivos</b> que se desarrollarán en el <b>Sprint 3</b>.
</p>
</div>
""").lstrip()  # lstrip por si quedó un \n inicial

if section == "Resumen Sprint 2":
    st.subheader("SPRINT 2 — Resumen General del Análisis")
    st.caption("Síntesis de los resultados obtenidos y aprendizajes clave del análisis de datos realizado en este módulo.")
    st.markdown("---")
    st.markdown(html_resumen_sprint2, unsafe_allow_html=True)

