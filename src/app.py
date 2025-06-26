
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
# Configuración de la página de Streamlit
st.set_page_config(page_title="Clasificador de Géneros Musicales", page_icon="🎵", layout="wide")

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models_mejorados"
DATA_DIR = ROOT_DIR / "data"

# --- FUNCIONES DE CARGA (CACHEADAS) ---
@st.cache_resource
def load_artifacts():
    """Carga el modelo, el escalador y el codificador de etiquetas. Se cachea para eficiencia."""
    model_path = MODELS_DIR / 'modelo_mejorado.joblib'
    scaler_path = MODELS_DIR / 'scaler_mejorado.joblib'
    encoder_path = MODELS_DIR / 'encoder_mejorado.joblib'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return model, scaler, label_encoder

@st.cache_data
def load_data():
    """
    Carga el dataset de características.
    CORRECCIÓN: El nombre del archivo CSV ahora coincide con el generado por '1_crear_dataset_mejorado.py'.
    """
    csv_path = DATA_DIR / 'features_refactored_3_sec.csv'
    df = pd.read_csv(csv_path)
    return df

# --- FUNCIONES DE PROCESAMIENTO DE AUDIO ---
def extract_features_from_audio(audio_bytes, sr=22050):
    """
    Extrae características de un archivo de audio subido.
    REFACTORIZACIÓN: Usa un diccionario para mayor claridad y consistencia,
    replicando la lógica del script '1_crear_dataset_mejorado.py'.
    """
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)

        # Para consistencia con los datos de entrenamiento, se analiza un segmento de 3 segundos del centro.
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
        tempo_array = librosa.beat.tempo(y=y, sr=sr)
        features['tempo'] = tempo_array[0] if isinstance(tempo_array, np.ndarray) and tempo_array.size > 0 else 0
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
    """Genera y muestra un gráfico de la forma de onda del audio."""
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
    # Obtenemos el orden correcto de las columnas para garantizar predicciones correctas.
    FEATURE_COLS = df.drop(columns=['filename', 'label']).columns
    LOAD_SUCCESS = True
except FileNotFoundError as e:
    st.error(f"Error CRÍTICO al cargar archivos: {e}")
    st.info(f"Asegúrate que la carpeta 'models_mejorados' y el archivo '{DATA_DIR / 'features_refactored_3_sec.csv'}' existen.")
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

    # --- Columna 1: Subir Archivo ---
    with col1:
        st.header("Opción 1: Sube tu archivo")
        uploaded_file = st.file_uploader("Selecciona un archivo de audio", type=["mp3", "wav", "au"], label_visibility="collapsed")

        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format=f'audio/{uploaded_file.type.split("/")[1]}')

            if st.button("🎤 Analizar y Clasificar", use_container_width=True):
                with st.spinner("Extrayendo características del audio..."):
                    features_df, y, sr = extract_features_from_audio(audio_bytes)

                if features_df is not None:
                    st.success("¡Análisis completado!")
                    with st.expander("Ver Forma de Onda del Audio"):
                        plot_waveform(y, sr)

                    # Aseguramos el orden correcto de las columnas antes de escalar
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

    # --- Columna 2: Usar Muestra ---
    with col2:
        st.header("Opción 2: Usar una muestra")
        df['song_base'] = df['filename'].apply(lambda x: '.'.join(x.split('.')[:-2]))
        song_list = sorted(df['song_base'].unique())
        selected_song_base = st.selectbox("Selecciona una canción de muestra:", options=song_list, label_visibility="collapsed")

        if st.button("🎶 Clasificar Muestra", use_container_width=True):
            song_to_classify_filename = f"{selected_song_base}.0.wav"
            song_data = df[df['filename'] == song_to_classify_filename]

            if not song_data.empty:
                song_features = song_data[FEATURE_COLS]
                features_scaled = scaler.transform(song_features)
                prediction_encoded = model.predict(features_scaled)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

                st.success(f"El género de la muestra es: **{prediction_label.capitalize()}**")
                
                genre_folder = song_data.iloc[0]['label']
                original_filename_wav = f"{selected_song_base}.wav"
                original_filename_au = f"{selected_song_base}.au"
                audio_path_wav = DATA_DIR / "genres_original" / genre_folder / original_filename_wav
                audio_path_au = DATA_DIR / "genres_original" / genre_folder / original_filename_au

                if audio_path_wav.is_file():
                    st.audio(audio_path_wav.read_bytes(), format='audio/wav')
                elif audio_path_au.is_file():
                    st.audio(audio_path_au.read_bytes(), format='audio/au')
                else:
                    st.warning(f"No se encontró el archivo de audio original para reproducir.")
            else:
                st.error(f"No se encontraron datos para la canción: '{song_to_classify_filename}'")
else:
    st.warning("La aplicación no puede funcionar hasta que se resuelvan los errores de carga de archivos.")
