# 1_crear_dataset_mejorado.py (Versión con barra de progreso y manejo de errores)

import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm # <-- MEJORA: Importamos la librería para la barra de progreso

# --- CONFIGURACIÓN ---
DATA_PATH = os.path.join('data', 'genres_original')
CSV_PATH = os.path.join('data', 'features_59_char_3_sec.csv')
SAMPLE_RATE = 22050
DURATION = 30 
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def main():
    """
    Función principal para extraer características de todos los archivos de audio
    y guardarlas en un nuevo archivo CSV. VERSIÓN CORREGIDA FINAL.
    """
    all_features = []
    genres = [genre for genre in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, genre))]
    
    for genre_label in tqdm(genres, desc="Procesando Géneros"):
        genre_path = os.path.join(DATA_PATH, genre_label)
        filenames = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        
        for f in tqdm(filenames, desc=f"  -> {genre_label}", leave=False):
            file_path = os.path.join(genre_path, f)
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                samples_per_segment = 3 * sr
                num_segments = 10
                
                for s in range(num_segments):
                    start_sample = samples_per_segment * s
                    finish_sample = start_sample + samples_per_segment
                    segment = signal[start_sample:finish_sample]
                    
                    if len(segment) == samples_per_segment:
                        features = []
                        
                        chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr); features.extend([np.mean(chroma_stft), np.var(chroma_stft)])
                        rms = librosa.feature.rms(y=segment); features.extend([np.mean(rms), np.var(rms)])
                        spec_cent = librosa.feature.spectral_centroid(y=segment, sr=sr); features.extend([np.mean(spec_cent), np.var(spec_cent)])
                        spec_bw = librosa.feature.spectral_bandwidth(y=segment, sr=sr); features.extend([np.mean(spec_bw), np.var(spec_bw)])
                        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr); features.extend([np.mean(rolloff), np.var(rolloff)])
                        zcr = librosa.feature.zero_crossing_rate(y=segment); features.extend([np.mean(zcr), np.var(zcr)])
                        y_harm, y_perc = librosa.effects.hpss(y=segment); features.extend([np.mean(y_harm), np.var(y_harm), np.mean(y_perc), np.var(y_perc)])
                        tempo_array = librosa.feature.rhythm.tempo(y=segment, sr=sr)
                        tempo = tempo_array[0] if tempo_array.size > 0 else 0
                        features.append(tempo)
                        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
                        for e in mfcc: features.extend([np.mean(e), np.var(e)])
                        spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=sr); features.extend([np.mean(spectral_contrast), np.var(spectral_contrast)])
                        
                        features.insert(0, f'{f.replace(".wav", "")}.{s}.wav')
                        features.append(genre_label)
                        all_features.append(features)
            
            except Exception as e:
                print(f"\nADVERTENCIA: Se omitió el archivo {file_path} por un error: {e}")
                continue

    # El resto de la función para crear el DataFrame y el CSV se mantiene igual...
    header = ['filename', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'spectral_rolloff_mean', 'spectral_rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo']
    for i in range(1, 21): header.append(f'mfcc{i}_mean'); header.append(f'mfcc{i}_var')
    header.extend(['spectral_contrast_mean', 'spectral_contrast_var'])
    header.append('label')
    df = pd.DataFrame(all_features, columns=header)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n\nProceso completado. Nuevo dataset guardado en: {CSV_PATH}")


if __name__ == '__main__':
    main()