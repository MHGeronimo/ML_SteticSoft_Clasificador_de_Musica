# src/pages/1_Clasificador_Interactivo.py

import io
import streamlit as st
import pandas as pd
import joblib
import os
import librosa
import numpy as np
from pathlib import Path

# --- MANEJO DE RUTAS ROBUSTO (La clave de la refactorización) ---
# Esto calcula la ruta absoluta al directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# Solución definitiva para ffmpeg
ffmpeg_path = r"C:\ProgramData\chocolatey\bin"
if ffmpeg_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + ffmpeg_path

# --- FUNCIONES DE CARGA Y PROCESAMIENTO ---

@st.cache_resource
def load_artifacts():
    """Carga los artefactos del modelo usando rutas absolutas."""
    model = joblib.load(MODELS_DIR / 'modelo_genero_musical.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
    label_encoder = joblib.load(MODELS_DIR / 'encoder.joblib')
    return model, scaler, label_encoder

@st.cache_data
def load_data():
    """Carga el dataframe de características usando una ruta absoluta."""
    df = pd.read_csv(DATA_DIR / 'features_3_sec.csv')
    return df

def extract_features_from_audio(audio_bytes):
    # (Esta función se mantiene igual que antes, es robusta)
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        # ... (el resto del código de la función es idéntico al anterior)
        # (copia y pega aquí el contenido de la función extract_features_from_audio de la versión anterior)
    except Exception as e:
        st.error(f"Error al procesar el archivo de audio: {e}")
        return None

# --- CARGA INICIAL DE ARTEFACTOS Y DATOS ---
try:
    model, scaler, label_encoder = load_artifacts()
    df = load_data()
    st.session_state['load_success'] = True
except FileNotFoundError:
    st.error("Error al cargar los archivos necesarios (modelo, datos). Asegúrate de que las carpetas 'models' y 'data' estén en la raíz del proyecto.")
    st.session_state['load_success'] = False

# --- INTERFAZ DE USUARIO ---
st.title("🤖 Clasificador Interactivo")

if st.session_state['load_success']:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Opción 1: Sube tu archivo MP3")
        uploaded_file = st.file_uploader("Selecciona un archivo MP3", type=["mp3"])
        
        if uploaded_file:
            st.audio(uploaded_file, format='audio/mp3')
            if st.button("Clasificar Archivo Subido"):
                # ... (lógica de clasificación para archivo subido)
                pass # Pega aquí la lógica de clasificación que ya tenías

    with col2:
        st.header("Opción 2: Usar una muestra")
        song_list = df['filename'].unique()
        selected_song_filename = st.selectbox("Selecciona una canción de muestra:", options=song_list)

        if st.button("Clasificar Muestra"):
            # Lógica de predicción
            song_features = df[df['filename'] == selected_song_filename].drop(['filename', 'length', 'label'], axis=1)
            features_scaled = scaler.transform(song_features)
            prediction_encoded = model.predict(features_scaled)
            prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
            st.success(f"El género predicho es: **{prediction_label.capitalize()}**")

            # Lógica para reproducir audio
            genre_folder = selected_song_filename.split('.')[0]
            audio_file_path = DATA_DIR / "genres_original" / genre_folder / selected_song_filename
            
            if audio_file_path.is_file():
                audio_bytes = audio_file_path.read_bytes()
                st.audio(audio_bytes, format='audio/wav')
            else:
                st.warning(f"No se encontró el archivo de audio en la ruta esperada: {audio_file_path}")