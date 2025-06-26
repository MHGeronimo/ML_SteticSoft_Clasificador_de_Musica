# src/app.py (Versión final con Opción 1 y Opción 2 funcionales)

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import io
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN Y RUTAS ---
st.set_page_config(page_title="Clasificador de Géneros Musicales", page_icon="🎵", layout="wide")

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models_mejorados"
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "audio_samples"

# --- FUNCIONES DE CARGA (CACHEADAS) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / 'modelo_mejorado.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler_mejorado.joblib')
    label_encoder = joblib.load(MODELS_DIR / 'encoder_mejorado.joblib')
    return model, scaler, label_encoder

@st.cache_data
def load_data():
    csv_path = DATA_DIR / 'features_refactored_3_sec.csv'
    df = pd.read_csv(csv_path)
    return df

# --- FUNCIONES DE PROCESAMIENTO DE AUDIO ---
def extract_features_from_audio(audio_bytes, sr=22050):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        if len(y) > sr * 3:
            start_sample = int((len(y) - sr * 3) / 2)
            y = y[start_sample : start_sample + sr * 3]

        features = {}
        features['chroma_stft_mean'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        features['chroma_stft_var'] = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
        features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
        features['rms_var'] = np.var(librosa.feature.rms(y=y))
        features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_centroid_var'] = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_bandwidth_var'] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['spectral_rolloff_var'] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=y))
        features['zero_crossing_rate_var'] = np.var(librosa.feature.zero_crossing_rate(y=y))
        y_harm, y_perc = librosa.effects.hpss(y=y)
        features['harmony_mean'] = np.mean(y_harm)
        features['harmony_var'] = np.var(y_harm)
        features['perceptr_mean'] = np.mean(y_perc)
        features['perceptr_var'] = np.var(y_perc)
        tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo_array[0] if isinstance(tempo_array, np.ndarray) and tempo_array.size > 0 else 0.0
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i, e in enumerate(mfcc, 1):
            features[f'mfcc{i}_mean'] = np.mean(e)
            features[f'mfcc{i}_var'] = np.var(e)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_var'] = np.var(spectral_contrast)
        
        features_df = pd.DataFrame([features])
        return features_df, y, sr
    except Exception as e:
        st.error(f"Error al procesar el archivo de audio: {e}")
        return None, None, None

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#6a0dad', alpha=0.7)
    ax.set_title('Forma de Onda del Audio Analizado')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# --- CARGA INICIAL Y VALIDACIÓN ---
try:
    model, scaler, label_encoder = load_artifacts()
    df = load_data()
    FEATURE_COLS = df.drop(columns=['filename', 'label']).columns
    LOAD_SUCCESS = True
except FileNotFoundError as e:
    st.error(f"Error CRÍTICO al cargar archivos: {e}")
    st.info(f"Asegúrate que las carpetas 'models_mejorados', 'data/audio_samples' y el archivo '{DATA_DIR / 'features_refactored_3_sec.csv'}' existan en el repositorio.")
    LOAD_SUCCESS = False
except Exception as e:
    st.error(f"Ocurrió un error inesperado durante la carga: {e}")
    LOAD_SUCCESS = False

# --- INTERFAZ DE USUARIO ---
st.title("🎵 Clasificador de Géneros Musicales")
st.markdown("Sube un archivo de audio (MP3, WAV) o elige una muestra para descubrir su género musical.")

if LOAD_SUCCESS:
    with st.expander("Ver los 10 géneros que el modelo puede identificar"):
        capitalized_genres = [genre.capitalize() for genre in label_encoder.classes_]
        st.info(", ".join(capitalized_genres))

    col1, col2 = st.columns(2, gap="large")

    # --- Opción 1: Subir Archivo ---
    with col1:
        st.header("Opción 1: Sube tu archivo")
        uploaded_file = st.file_uploader("Selecciona un archivo de audio", type=["mp3", "wav", "au"], label_visibility="collapsed")
        
        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format=f'audio/{uploaded_file.type.split("/")[1]}')
            
            # --- CORRECCIÓN: SE RESTAURÓ LA LÓGICA DE CLASIFICACIÓN AQUÍ ---
            if st.button("🎤 Analizar y Clasificar", use_container_width=True, key="classify_uploaded"):
                with st.spinner("Extrayendo características del audio..."):
                    features_df, y, sr = extract_features_from_audio(audio_bytes)

                if features_df is not None:
                    st.success("¡Análisis completado!")
                    with st.expander("Ver Forma de Onda del Audio"):
                        plot_waveform(y, sr)
                    
                    features_df_ordered = features_df[FEATURE_COLS]
                    features_scaled = scaler.transform(features_df_ordered)

                    st.subheader("🏆 Ranking de Géneros Probables")
                    probabilities = model.predict_proba(features_scaled)[0]
                    prob_df = pd.DataFrame({'Género': label_encoder.classes_, 'Confianza': probabilities})
                    top_genres = prob_df.sort_values(by='Confianza', ascending=False).reset_index(drop=True)

                    main_genre = top_genres.iloc[0]
                    st.metric(label="Género Principal", value=main_genre['Género'].capitalize(), delta=f"{main_genre['Confianza']:.1%} de confianza")

                    if len(top_genres) > 1:
                        st.write("**Posibles géneros secundarios:**")
                        for _, row in top_genres.iloc[1:3].iterrows():
                            st.write(f"- **{row['Género'].capitalize()}** con {row['Confianza']:.1%} de confianza.")

                    with st.expander("Ver desglose de confianza para todos los géneros"):
                        st.bar_chart(prob_df.set_index('Género'))

    # --- Opción 2: Usar Muestra ---
    with col2:
        st.header("Opción 2: Usar una muestra")
        sample_genres = sorted(label_encoder.classes_)
        selected_genre = st.selectbox("Selecciona un género de muestra:", options=sample_genres)

        if st.button("🎶 Clasificar Muestra", use_container_width=True, key="classify_sample"):
            sample_audio_filename = f"{selected_genre}.00000.wav"
            sample_audio_path = SAMPLES_DIR / sample_audio_filename
            sample_feature_filename = f"{selected_genre}.00000.0.wav"
            song_data = df[df['filename'] == sample_feature_filename]

            if not song_data.empty and sample_audio_path.is_file():
                song_features = song_data[FEATURE_COLS]
                features_scaled = scaler.transform(song_features)
                prediction_encoded = model.predict(features_scaled)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

                st.success(f"El género predicho para la muestra de '{selected_genre.capitalize()}' es: **{prediction_label.capitalize()}**")
                
                audio_bytes = sample_audio_path.read_bytes()
                st.audio(audio_bytes, format='audio/wav')
                
            elif not sample_audio_path.is_file():
                st.error(f"No se encontró el archivo de muestra: '{sample_audio_filename}' en la carpeta 'data/audio_samples'.")
            else:
                st.error(f"No se encontraron datos en el CSV para la canción: '{sample_feature_filename}'")
else:
    st.warning("La aplicación no puede funcionar hasta que se resuelvan los errores de carga de archivos.")
