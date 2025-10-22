
# Tienda Aurelion - App Interactiva (Streamlit)
# ------------------------------------------------
# Requisitos:
#   pip install streamlit pandas openpyxl
#
# Ejecutar:
#   streamlit run streamlit_aurelion_app_logo_fixed.py
#
# Estructura esperada:
#   AURELION/
#   ├── streamlit_aurelion_app_logo_fixed.py
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
# Navegación lateral
# --------------------------------------
st.sidebar.title("Navegación")
section = st.sidebar.radio("Ir a:", ["Temas", "Fuentes", "Pseudocódigo", "Diagrama", "Resumen Sprint 1"])


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
