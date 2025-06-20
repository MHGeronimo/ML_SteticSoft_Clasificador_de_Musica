# app.py (Versión final usando el MODELO MEJORADO)

# 1. IMPORTACIONES
import streamlit as st
import pandas as pd
import joblib
import os
import librosa
import librosa.display
import numpy as np
import io
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Clasificador de Géneros Musicales",
    page_icon="🎵",
    layout="wide"
)

# --- FUNCIONES ---
@st.cache_resource
def load_artifacts():
    """
    --- MEJORA 1: Carga los artefactos del modelo MEJORADO ---
    """
    model = joblib.load(os.path.join('models_mejorados', 'modelo_mejorado.joblib'))
    scaler = joblib.load(os.path.join('models_mejorados', 'scaler_mejorado.joblib'))
    label_encoder = joblib.load(os.path.join('models_mejorados', 'encoder_mejorado.joblib'))
    return model, scaler, label_encoder

@st.cache_data
def load_data():
    # Carga el nuevo CSV con 59 características para la Opción 2
    df = pd.read_csv(os.path.join('data', 'features_59_char_3_sec.csv'))
    return df

def extract_features_from_audio(audio_bytes):
    """
    Extrae las 57 características de un archivo de audio en memoria.
    VERSIÓN CORREGIDA para compatibilidad de librosa.
    """
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # Características
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        y_harm, y_perc = librosa.effects.hpss(y)

        # --- INICIO DE LA CORRECCIÓN DE VERSIÓN ---
        # Revertimos al uso de librosa.beat.tempo, que es compatible con tu versión instalada.
        # La advertencia 'FutureWarning' puede volver a aparecer en la consola, pero es inofensiva.
        tempo_array = librosa.beat.tempo(y=y, sr=sr)
        tempo = tempo_array[0] if tempo_array.size > 0 else 0
        # --- FIN DE LA CORRECCIÓN DE VERSIÓN ---

        # Construcción del vector de características
        features_list = [
            np.mean(chroma_stft), np.var(chroma_stft),
            np.mean(rms), np.var(rms),
            np.mean(spec_cent), np.var(spec_cent),
            np.mean(spec_bw), np.var(spec_bw),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zcr), np.var(zcr),
            tempo,
            np.mean(y_harm), np.var(y_harm),
            np.mean(y_perc), np.var(y_perc)
        ]
        
        for e in mfcc:
            features_list.extend([np.mean(e), np.var(e)])
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features_list.extend([np.mean(spectral_contrast), np.var(spectral_contrast)])

        return np.array(features_list).reshape(1, -1), y, sr

    except Exception as e:
        st.error(f"Error al procesar el archivo de audio: {e}")
        return None, None, None

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='purple', alpha=0.7)
    ax.set_title('Forma de Onda del Audio'); ax.set_xlabel('Tiempo (s)'); ax.set_ylabel('Amplitud')
    st.pyplot(fig)

try:
    model, scaler, label_encoder = load_artifacts()
    df = load_data()
    LOAD_SUCCESS = True
except FileNotFoundError:
    st.error("Error CRÍTICO: No se encontraron los archivos del modelo mejorado. Asegúrate de haber ejecutado '2_entrenar_modelo_mejorado.py' y que la carpeta 'models_mejorados' exista.")
    LOAD_SUCCESS = False
    
st.title("🎵 Clasificador de Géneros Musicales (Modelo v2 - Mejorado)")
st.write("Esta versión utiliza un modelo mejorado con 59 características para una mayor precisión.")

if LOAD_SUCCESS:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Opción 1: Sube tu archivo MP3")
        uploaded_file = st.file_uploader("Selecciona un archivo MP3", type=["mp3"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/mp3')
            if st.button("Analizar y Clasificar Archivo"):
                with st.spinner("Analizando audio y extrayendo características..."):
                    features, y, sr = extract_features_from_audio(audio_bytes)
                if features is not None:
                    st.success("Análisis completado. Estos son los resultados:")
                    with st.expander("Ver Forma de Onda del Audio"):
                        plot_waveform(y, sr)
                    features_scaled = scaler.transform(features)
                    prediction_encoded = model.predict(features_scaled)
                    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
                    st.metric(label="Género Predicho", value=prediction_label.capitalize())
                    st.write("**Confianza de la Predicción por Género:**")
                    probabilities = model.predict_proba(features_scaled)
                    prob_df = pd.DataFrame(probabilities, columns=label_encoder.classes_).T * 100
                    prob_df.columns = ["Confianza (%)"]
                    st.bar_chart(prob_df)

    with col2:
        st.header("Opción 2: Usar una muestra del Dataset")
        song_list = df['filename'].unique()
        selected_song_filename_from_csv = st.selectbox("Selecciona una canción:", options=song_list, label_visibility="collapsed")

        if st.button("Clasificar Muestra"):
            # Ahora usamos el nuevo dataframe de 59 características
            song_features = df[df['filename'] == selected_song_filename_from_csv].drop(['filename', 'label'], axis=1)
            features_scaled = scaler.transform(song_features)
            prediction_encoded = model.predict(features_scaled)
            prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
            st.success(f"El género predicho es: **{prediction_label.capitalize()}**")

            parts = selected_song_filename_from_csv.split('.')
            actual_filename = f"{parts[0]}.{parts[1]}.{parts[-1]}"
            genre_folder = actual_filename.split('.')[0]
            audio_file_path = os.path.join('data', 'genres_original', genre_folder, actual_filename)
            try:
                with open(audio_file_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
            except FileNotFoundError:
                st.warning(f"No se encontró el archivo de audio para reproducir.")