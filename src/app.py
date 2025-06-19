# src/app.py

# 1. Importaciones
import streamlit as st
import pandas as pd
import joblib
import os
import librosa
import numpy as np
from pydub import AudioSegment
import io

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Clasificador de Géneros Musicales",
    page_icon="🎵",
    layout="wide"
)

# --- FUNCIONES ---

@st.cache_resource
def load_artifacts():
    # Carga los modelos y preprocesadores
    model = joblib.load(os.path.join('models', 'modelo_genero_musical.joblib'))
    scaler = joblib.load(os.path.join('models', 'scaler.joblib'))
    label_encoder = joblib.load(os.path.join('models', 'encoder.joblib'))
    return model, scaler, label_encoder

def extract_features_from_audio(audio_bytes):
    """
    Extrae las 57 características de un archivo de audio en memoria.
    Replica el proceso usado para crear el dataset original.
    """
    try:
        # Usar un buffer de bytes en memoria
        audio_io = io.BytesIO(audio_bytes)
        
        # Cargar el audio con librosa
        y, sr = librosa.load(audio_io, sr=22050)

        # Dividir el audio en segmentos de 3 segundos
        segment_length = 3 * sr
        segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]

        all_features = []
        for segment in segments:
            if len(segment) == segment_length:
                # Extraer cada característica
                chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr)
                rms = librosa.feature.rms(y=segment)
                spec_cent = librosa.feature.spectral_centroid(y=segment, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y=segment)
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
                
                # Calcular media y varianza de cada característica
                features = [
                    np.mean(chroma_stft), np.var(chroma_stft),
                    np.mean(rms), np.var(rms),
                    np.mean(spec_cent), np.var(spec_cent),
                    np.mean(spec_bw), np.var(spec_bw),
                    np.mean(rolloff), np.var(rolloff),
                    np.mean(zcr), np.var(zcr)
                ]
                # Añadir media y varianza de los 20 MFCCs
                for e in mfcc:
                    features.append(np.mean(e))
                    features.append(np.var(e))
                
                all_features.append(features)

        if not all_features:
            return None

        # Promediar las características de todos los segmentos para obtener un vector único
        final_features = np.mean(all_features, axis=0)
        return final_features.reshape(1, -1)

    except Exception as e:
        st.error(f"Error al procesar el archivo de audio: {e}")
        return None


# --- CARGA DE DATOS Y MODELOS ---
model, scaler, label_encoder = load_artifacts()

# --- INTERFAZ DE USUARIO ---
st.title("🎵 Clasificador de Géneros Musicales")
st.write(
    "Sube tu propio archivo de audio en formato MP3 para que el modelo lo clasifique, "
    "o utiliza una de las canciones de muestra del dataset original."
)

# Colocar el cargador de archivos y el selector en columnas
col1, col2 = st.columns(2)

with col1:
    st.header("Opción 1: Sube tu archivo MP3")
    uploaded_file = st.file_uploader("Selecciona un archivo MP3", type=["mp3"])
    
    if uploaded_file is not None:
        # Leer los bytes del archivo
        audio_bytes = uploaded_file.read()
        
        # Mostrar reproductor de audio
        st.audio(audio_bytes, format='audio/mp3')

        if st.button("Clasificar Archivo Subido"):
            with st.spinner("Analizando el audio... Esto puede tardar unos segundos."):
                # Extraer características del archivo subido
                features = extract_features_from_audio(audio_bytes)
                
                if features is not None:
                    # Escalar y predecir
                    features_scaled = scaler.transform(features)
                    prediction_encoded = model.predict(features_scaled)
                    prediction_label = label_encoder.inverse_transform(prediction_encoded)
                    
                    st.success(f"El género predicho es: **{prediction_label[0].capitalize()}**")

with col2:
    st.header("Opción 2: Usar una muestra")
    # Lógica anterior para usar canciones de muestra (si se desea mantener)
    df = pd.read_csv(os.path.join('data', 'features_3_sec.csv'))
    song_list = df['filename'].unique()
    selected_song_filename = st.selectbox(
        "Selecciona una canción de muestra:",
        options=song_list
    )
    if st.button("Clasificar Muestra"):
        song_features = df[df['filename'] == selected_song_filename].drop(['filename', 'length', 'label'], axis=1)
        song_features_scaled = scaler.transform(song_features)
        prediction_encoded = model.predict(song_features_scaled)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)
        st.success(f"El género predicho es: **{prediction_label[0].capitalize()}**")