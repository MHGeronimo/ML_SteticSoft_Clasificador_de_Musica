# src/pages/1_Guía_de_Uso.py

import streamlit as st

st.set_page_config(page_title="Guía de Uso", page_icon="📖")

st.title("📖 Bienvenido y Guía de Uso")

st.markdown("""
Esta aplicación te permite clasificar el género de canciones usando un modelo de Machine Learning.

### ¿Cómo Usar la Aplicación?

1.  **Ve a la página principal "app" en el menú lateral.**
    - Allí encontrarás el clasificador interactivo.

2.  **Tienes dos opciones para clasificar:**
    - **Opción 1:** Sube tu propio archivo de audio en formato MP3. La aplicación lo analizará, te mostrará su forma de onda y predecirá su género junto con un gráfico de confianza.
    - **Opción 2:** Elige una de las canciones de muestra del dataset original y la aplicación la clasificará y reproducirá para ti.

3.  **Explora las otras páginas:**
    - **Análisis del Modelo:** Descubre qué tan preciso es nuestro modelo con métricas detalladas y gráficos.
    - **Sobre el Proyecto:** Conoce al equipo detrás del proyecto y las tecnologías que utilizamos.

¡Esperamos que disfrutes la experiencia!
""")

st.info("👈 Para empezar, selecciona **'app'** en el menú de la izquierda.", icon="ℹ️")