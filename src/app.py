# src/app.py (El Clasificador Principal con las mejoras de UX)

import streamlit as st
import pandas as pd
import joblib
import os
import librosa
import librosa.display
import numpy as np
import io
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURACIÓN Y RUTAS ---
st.set_page_config(page_title="Clasificador Interactivo", page_icon="🤖", layout="wide")

# Lógica de rutas robusta para app.py DENTRO de la carpeta 'src'
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models_mejorados"
DATA_DIR = ROOT_DIR / "data"

# --- FUNCIONES ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / 'modelo_mejorado.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler_mejorado.joblib')
    label_encoder = joblib.load(MODELS_DIR / 'encoder_mejorado.joblib')
    return model, scaler, label_encoder

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / 'features_59_char_3_sec.csv')
    return df

def extract_features_from_audio(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        tempo_array = librosa.beat.tempo(y=y, sr=sr); tempo = tempo_array[0] if tempo_array.size > 0 else 0
        y_harm, y_perc = librosa.effects.hpss(y);
        features_list = [np.mean(librosa.feature.chroma_stft(y=y, sr=sr)), np.var(librosa.feature.chroma_stft(y=y, sr=sr)), np.mean(librosa.feature.rms(y=y)), np.var(librosa.feature.rms(y=y)), np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)), np.var(librosa.feature.spectral_centroid(y=y, sr=sr)), np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)), np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)), np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)), np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)), np.mean(librosa.feature.zero_crossing_rate(y)), np.var(librosa.feature.zero_crossing_rate(y)), np.mean(y_harm), np.var(y_harm), np.mean(y_perc), np.var(y_perc), tempo]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20);
        for e in mfcc: features_list.extend([np.mean(e), np.var(e)])
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr); features_list.extend([np.mean(spectral_contrast), np.var(spectral_contrast)])
        return np.array(features_list).reshape(1, -1), y, sr
    except Exception as e:
        st.error(f"Error al procesar el archivo de audio: {e}"); return None, None, None

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3)); librosa.display.waveshow(y, sr=sr, ax=ax, color='purple', alpha=0.7); ax.set_title('Forma de Onda'); ax.set_xlabel('Tiempo (s)'); ax.set_ylabel('Amplitud'); st.pyplot(fig)

# --- CARGA INICIAL ---
try:
    model, scaler, label_encoder = load_artifacts()
    df = load_data()
    LOAD_SUCCESS = True
except FileNotFoundError:
    st.error(f"Error CRÍTICO al cargar archivos. Asegúrate que las carpetas 'models_mejorados' y 'data' ({DATA_DIR}) estén en la raíz del proyecto.")
    LOAD_SUCCESS = False

# --- INTERFAZ DE USUARIO ---
st.title("🤖 Clasificador Interactivo de Música")

if LOAD_SUCCESS:
    # --- MEJORA 1: Lista de géneros soportados ---
    with st.expander("Ver los 10 géneros que el modelo puede identificar"):
        st.info(", ".join(label_encoder.classes_).capitalize())

    col1, col2 = st.columns(2)

    with col1:
        st.header("Opción 1: Sube tu archivo MP3")
        uploaded_file = st.file_uploader("Selecciona un archivo MP3", type=["mp3"], label_visibility="collapsed")
        
        if uploaded_file:
            audio_bytes = uploaded_file.read(); st.audio(audio_bytes, format='audio/mp3')
            if st.button("Analizar y Clasificar Archivo"):
                with st.spinner("Analizando audio..."):
                    features, y, sr = extract_features_from_audio(audio_bytes)
                if features is not None:
                    st.success("Análisis completado:")
                    with st.expander("Ver Forma de Onda"): plot_waveform(y, sr)
                    
                    features_scaled = scaler.transform(features)
                    
                    # --- MEJORA 2: Ranking de Géneros ---
                    st.subheader("🏆 Ranking de Géneros Probables")
                    probabilities = model.predict_proba(features_scaled)[0]
                    prob_df = pd.DataFrame({'Género': label_encoder.classes_, 'Confianza': probabilities})
                    top_3_genres = prob_df.sort_values(by='Confianza', ascending=False).head(3)
                    
                    st.metric(label="Género Principal (Más probable)", value=top_3_genres.iloc[0]['Género'].capitalize(), delta=f"{top_3_genres.iloc[0]['Confianza']:.1%} de confianza")
                    
                    if len(top_3_genres) > 1:
                        st.write("**Géneros Secundarios:**")
                        for i, row in top_3_genres.iloc[1:].iterrows():
                            st.write(f"- **{row['Género'].capitalize()}** con un {row['Confianza']:.1%} de confianza.")
                    
                    with st.expander("Ver confianza para todos los géneros"):
                        st.bar_chart(prob_df.set_index('Género'))

    with col2:
        st.header("Opción 2: Usar una muestra")
        song_list = df['filename'].unique()
        selected_song_filename_from_csv = st.selectbox("Selecciona una canción de muestra:", options=song_list, label_visibility="collapsed")
        if st.button("Clasificar Muestra"):
            song_features = df[df['filename'] == selected_song_filename_from_csv].drop(['filename', 'label'], axis=1)
            features_scaled = scaler.transform(song_features)
            prediction_encoded = model.predict(features_scaled)
            prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
            st.success(f"El género predicho es: **{prediction_label.capitalize()}**")
            parts = selected_song_filename_from_csv.split('.'); actual_filename = f"{parts[0]}.{parts[1]}.{parts[-1]}"
            genre_folder = actual_filename.split('.')[0]
            audio_file_path = DATA_DIR / "genres_original" / genre_folder / actual_filename
            if audio_file_path.is_file():
                audio_bytes = audio_file_path.read_bytes()
                st.audio(audio_bytes, format='audio/wav')
            else:
                st.warning(f"No se encontró el audio para reproducir.")
else:
    st.error("La aplicación no puede funcionar.")