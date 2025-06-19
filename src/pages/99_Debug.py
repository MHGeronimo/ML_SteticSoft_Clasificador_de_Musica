# src/pages/99_Debug.py

import streamlit as st
from pathlib import Path
import os

st.set_page_config(page_title="Página de Depuración", layout="wide")

st.title("🕵️‍♂️ Página de Depuración de Rutas")

st.info(
    "Esta página nos ayuda a verificar si el programa puede 'ver' los archivos y carpetas "
    "en las ubicaciones correctas."
)

# --- 1. VERIFICAR RUTA RAÍZ Y CARPETAS PRINCIPALES ---
st.header("1. Verificación de Rutas Principales")

try:
    # __file__ se refiere a este mismo script (99_Debug.py)
    # .parent es la carpeta que lo contiene (pages)
    # .parent.parent es la carpeta que contiene a 'pages' (src)
    # .parent.parent.parent es la carpeta raíz del proyecto
    ROOT_DIR = Path(__file__).parent.parent.parent
    st.write(f"**Ruta Raíz del Proyecto Calculada:** `{ROOT_DIR}`")

    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "models"

    st.write(f"**Buscando la carpeta 'data' en:** `{DATA_DIR}`")
    if DATA_DIR.exists() and DATA_DIR.is_dir():
        st.success("✔️ Carpeta 'data' encontrada.")
    else:
        st.error("❌ ¡ERROR! No se encontró la carpeta 'data' en la ruta esperada.")

    st.write(f"**Buscando la carpeta 'models' en:** `{MODELS_DIR}`")
    if MODELS_DIR.exists() and MODELS_DIR.is_dir():
        st.success("✔️ Carpeta 'models' encontrada.")
    else:
        st.error("❌ ¡ERROR! No se encontró la carpeta 'models' en la ruta esperada.")

except Exception as e:
    st.error(f"Ocurrió un error al calcular las rutas base: {e}")


# --- 2. VERIFICAR CARPETA DE AUDIO Y UN ARCHIVO ESPECÍFICO ---
st.header("2. Verificación de Archivos de Audio")

if 'DATA_DIR' in locals() and DATA_DIR.exists():
    GENRES_DIR = DATA_DIR / "genres_original"
    st.write(f"**Buscando la carpeta de géneros en:** `{GENRES_DIR}`")

    if GENRES_DIR.exists() and GENRES_DIR.is_dir():
        st.success("✔️ Carpeta 'genres_original' encontrada dentro de 'data'.")

        # Prueba con un archivo específico
        BLUES_DIR = GENRES_DIR / "blues"
        st.write(f"**Buscando la sub-carpeta 'blues' en:** `{BLUES_DIR}`")
        if BLUES_DIR.exists():
             st.success("✔️ Sub-carpeta 'blues' encontrada.")
        else:
            st.error("❌ ¡ERROR! No se encontró la sub-carpeta 'blues' dentro de 'genres_original'.")

        # Prueba final con un archivo .wav
        TEST_FILE_PATH = BLUES_DIR / "blues.00000.wav"
        st.write(f"**Buscando el archivo de prueba en:** `{TEST_FILE_PATH}`")
        if TEST_FILE_PATH.is_file():
            st.success("✔️ ¡ÉXITO! El archivo 'blues.00000.wav' fue encontrado y es accesible.")
        else:
            st.error("❌ ¡ERROR CRÍTICO! El archivo 'blues.00000.wav' no se encontró en la ruta final.")
    else:
        st.error("❌ ¡ERROR! No se encontró la carpeta 'genres_original' dentro de 'data'.")