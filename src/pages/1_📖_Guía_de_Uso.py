# src/pages/1_Guía_de_Uso.py (Versión con imagen local)

import streamlit as st
from pathlib import Path # <-- Importamos Path

st.set_page_config(page_title="Guía de Uso", page_icon="📖")

# --- Lógica para encontrar la imagen local ---
ROOT_DIR = Path(__file__).parent.parent.parent
IMAGE_PATH = ROOT_DIR / "assets" / "welcome_image.jpg"

st.title("📖 Bienvenido al Clasificador Musical")

# --- Mostrar la imagen local ---
if IMAGE_PATH.is_file():
    st.image(str(IMAGE_PATH), width=600) # st.image necesita la ruta como texto
else:
    st.warning("No se encontró la imagen de bienvenida en la carpeta 'assets'.")


st.markdown("""
Esta aplicación te permite clasificar el género de canciones usando un modelo de Machine Learning.

### ¿Cómo Usar la Aplicación?

1.  **Ve a la página principal "app" en el menú lateral.**
    - Allí encontrarás el clasificador interactivo para probar el modelo.

2.  **Explora las otras páginas:**
    - **Análisis del Modelo:** Descubre qué tan preciso es nuestro modelo con métricas detalladas.
    - **Sobre el Proyecto:** Conoce al equipo y las tecnologías que utilizamos.
    - **Explorar Canciones:** Descubre ejemplos de canciones para cada uno de los 10 géneros.

¡Esperamos que disfrutes la experiencia!
""")
st.info("👈 Para empezar, selecciona **'app'** en el menú de la izquierda.", icon="ℹ️")