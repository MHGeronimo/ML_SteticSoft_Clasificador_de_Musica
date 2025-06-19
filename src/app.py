# app.py (Versión con la función de extracción de características CORREGIDA)

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
    model = joblib.load(os.path.join('models', 'modelo_genero_musical.joblib'))
    scaler = joblib.load(os.path.join('models', 'scaler.joblib'))
    label_encoder = joblib.load(os.path.join('models', 'encoder.joblib'))
    return model, scaler, label_encoder

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join('data', 'features_3_sec.csv'))
    return df

def extract_features_from_audio(audio_bytes):
    """
    Extrae las 57 características de un archivo de audio en memoria.
    VERSIÓN CORREGIDA para manejar el tempo correctamente.
    """
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # --- Características que se mantienen ---
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        y_harm, y_perc = librosa.effects.hpss(y)

        # --- INICIO DE LA CORRECCIÓN ---
        # Se usa librosa.beat.tempo que es más directo y nos aseguramos de tomar solo un valor.
        tempo_array = librosa.beat.tempo(y=y, sr=sr)
        tempo = tempo_array[0] if tempo_array.size > 0 else 0 # Tomamos el primer valor del array
        # --- FIN DE LA CORRECCIÓN ---

        # Construcción del vector de características (ahora todos son números únicos)
        features_list = [
            np.mean(chroma_stft), np.var(chroma_stft),
            np.mean(rms), np.var(rms),
            np.mean(spec_cent), np.var(spec_cent),
            np.mean(spec_bw), np.var(spec_bw),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zcr), np.var(zcr),
            tempo, # <-- Ahora es garantizado que es un número único
            np.mean(y_harm), np.var(y_harm),
            np.mean(y_perc), np.var(y_perc)
        ]
        
        for e in mfcc:
            features_list.extend([np.mean(e), np.var(e)])
        
        return np.array(features_list).reshape(1, -1), y, sr

    except Exception as e:
        st.error(f"Error al procesar el archivo de audio: {e}")
        return None, None, None

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='purple', alpha=0.7)
    ax.set_title('Forma de Onda del Audio')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud')
    st.pyplot(fig)

# --- CARGA DE DATOS Y MODELOS ---
try:
    model, scaler, label_encoder = load_artifacts()
    df = load_data()
    LOAD_SUCCESS = True
except FileNotFoundError:
    st.error("Error CRÍTICO al cargar archivos. Asegúrate que las carpetas 'models' y 'data' estén en la raíz del proyecto.")
    LOAD_SUCCESS = False
    
# --- INTERFAZ DE USUARIO ---
st.title("🎵 Clasificador de Géneros Musicales")
st.write("Sube un archivo de audio en formato MP3 para que el modelo lo clasifique, o utiliza una de las canciones de muestra del dataset original.")

if LOAD_SUCCESS:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Opción 1: Sube tu archivo MP3")
        uploaded_file = st.file_uploader("Selecciona un archivo MP3", type=["mp3"])
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/mp3')

            if st.button("Clasificar Archivo Subido"):
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
        st.header("Opción 2: Usar una muestra")
        song_list = df['filename'].unique()
        selected_song_filename_from_csv = st.selectbox("Selecciona una canción de muestra:", options=song_list)

        if st.button("Clasificar Muestra"):
            song_features = df[df['filename'] == selected_song_filename_from_csv].drop(['filename', 'length', 'label'], axis=1)
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