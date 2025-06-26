import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
import warnings

# --- CONFIGURACIÓN ---
DATA_PATH = os.path.join('data', 'genres_original')
CSV_PATH = os.path.join('data', 'features_refactored_3_sec.csv')
SAMPLE_RATE = 22050
NUM_SEGMENTS = 10
SAMPLES_PER_SEGMENT = int((SAMPLE_RATE * 30) / NUM_SEGMENTS)


def extract_features(segment, sr, segment_num, original_filename):
    """
    Extrae un conjunto de características de un segmento de audio y las devuelve en un diccionario.
    """
    if len(segment) != SAMPLES_PER_SEGMENT:
        return None

    features = {}

    # Metadata
    features['filename'] = f'{os.path.splitext(original_filename)[0]}.{segment_num}.wav'

    # Extracción de características de Librosa
    features['chroma_stft_mean'] = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr))
    features['chroma_stft_var'] = np.var(librosa.feature.chroma_stft(y=segment, sr=sr))
    features['rms_mean'] = np.mean(librosa.feature.rms(y=segment))
    features['rms_var'] = np.var(librosa.feature.rms(y=segment))
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
    features['spectral_centroid_var'] = np.var(librosa.feature.spectral_centroid(y=segment, sr=sr))
    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
    features['spectral_bandwidth_var'] = np.var(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
    features['spectral_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))
    features['spectral_rolloff_var'] = np.var(librosa.feature.spectral_rolloff(y=segment, sr=sr))
    features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=segment))
    features['zero_crossing_rate_var'] = np.var(librosa.feature.zero_crossing_rate(y=segment))
    y_harm, y_perc = librosa.effects.hpss(y=segment)
    features['harmony_mean'] = np.mean(y_harm)
    features['harmony_var'] = np.var(y_harm)
    features['perceptr_mean'] = np.mean(y_perc)
    features['perceptr_var'] = np.var(y_perc)
    
    # --- CORRECCIÓN FINAL ---
    # Se actualiza la función de tempo a la recomendada por librosa para evitar el FutureWarning.
    tempo_array, _ = librosa.beat.beat_track(y=segment, sr=sr)
    features['tempo'] = tempo_array[0] if isinstance(tempo_array, np.ndarray) and tempo_array.size > 0 else 0.0

    # MFCCs
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
    for i, e in enumerate(mfcc, 1):
        features[f'mfcc{i}_mean'] = np.mean(e)
        features[f'mfcc{i}_var'] = np.var(e)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_var'] = np.var(spectral_contrast)

    return features


def create_dataset():
    """
    Función principal para extraer características de todos los archivos de audio
    y guardarlas en un nuevo archivo CSV.
    """
    warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
    all_features_list = []

    try:
        genres = [g for g in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, g))]
        if not genres:
            print(f"Error: No se encontraron directorios de géneros en la ruta '{DATA_PATH}'.")
            return
    except FileNotFoundError:
        print(f"Error Crítico: El directorio de datos '{DATA_PATH}' no fue encontrado.")
        print("Asegúrate de que la estructura de carpetas 'data/genres_original/<genero>/...' es correcta.")
        return

    for genre_label in tqdm(genres, desc="Procesando Géneros"):
        genre_path = os.path.join(DATA_PATH, genre_label)
        filenames = [f for f in os.listdir(genre_path) if f.lower().endswith(('.wav', '.au'))]

        for f in tqdm(filenames, desc=f"  -> {genre_label:10s}", leave=False):
            file_path = os.path.join(genre_path, f)
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                for s in range(NUM_SEGMENTS):
                    start = SAMPLES_PER_SEGMENT * s
                    finish = start + SAMPLES_PER_SEGMENT
                    segment = signal[start:finish]
                    features = extract_features(segment, sr, s, f)
                    if features:
                        features['label'] = genre_label
                        all_features_list.append(features)
            except Exception as e:
                print(f"\nADVERTENCIA: Se omitió el archivo {file_path} durante la carga. Error: {e}")
                continue

    if not all_features_list:
        print("\nNo se extrajo ninguna característica. El proceso finalizó sin crear un archivo CSV.")
        return

    df = pd.DataFrame(all_features_list)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n\nProceso completado. Dataset guardado en: {CSV_PATH}")
    print(f"Se procesaron un total de {len(df)} segmentos.")


if __name__ == '__main__':
    create_dataset()
