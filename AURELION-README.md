
# 🛒 Tienda Aurelion — App Interactiva (Streamlit)

Aplicación desarrollada en **Python + Streamlit** para visualizar y analizar datos de clientes, ventas, productos y métodos de pago.

---

## 📁 Estructura del proyecto

```
AURELION/
├── aurelion_app.py
├── requirements.txt
├── BD/
│   ├── clientes.xlsx
│   ├── productos.xlsx
│   ├── ventas.xlsx
│   └── detalle_ventas.xlsx
└── IMAGES/
    ├── LOGO.png
    └── LOGO2.png
```

---

## ⚙️ Instalación de dependencias

Desde una terminal dentro de la carpeta **AURELION**, ejecutar:

```bash
pip install streamlit pandas openpyxl graphviz
```

> 💡 *El paquete `graphviz` es opcional. Solo se usa para visualizar el diagrama de flujo.*

---

## ▶️ Ejecución de la app

Ejecutar el siguiente comando:

```bash
streamlit run aurelion_app.py
```

La aplicación se abrirá automáticamente en el navegador predeterminado.

---

## 📊 Descripción general

La app permite navegar entre secciones:

- **Temas:** Presenta los problemas y soluciones propuestos.
- **Fuentes:** Muestra la estructura y tipos de datos de cada dataset.
- **Pseudocódigo:** Resume los pasos lógicos aplicados a cada tema.
- **Diagrama:** Visualiza el flujo general de la aplicación.
- **Resumen Sprint 1:** Describe los entregables completados y los próximos pasos.

---

## 🧩 Requisitos técnicos

- Python 3.9 o superior  
- Librerías: `streamlit`, `pandas`, `openpyxl`, *(opcionalmente)* `graphviz`

---

## ✨ Autor

**Leandro Serantes**  
Proyecto  — *Tienda Aurelion (Sprint 1)*  
