# 2_entrenar_modelo_mejorado.py (Versión robusta y con mejor feedback)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys # Importamos sys para poder detener el script de forma controlada

def main():
    """
    Función principal para entrenar el modelo mejorado, ahora más robusta.
    """
    CSV_PATH = os.path.join('data', 'features_59_char_3_sec.csv')
    MODELS_DIR = 'models_mejorados'

    print("--- INICIANDO PROCESO DE ENTRENAMIENTO DEL MODELO MEJORADO ---")

    # --- MEJORA 1: Manejo de Errores y Validación de Datos ---
    print("\n[Fase 1 de 4] Cargando y validando el dataset...")
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Dataset '{CSV_PATH}' cargado exitosamente.")
    except FileNotFoundError:
        print(f"\n[ERROR CRÍTICO] No se encontró el archivo del dataset en la ruta: '{CSV_PATH}'")
        print("Por favor, asegúrate de haber ejecutado primero el script '1_crear_dataset_mejorado.py'.")
        sys.exit() # Detiene la ejecución si no encuentra el archivo

    # Validar datos nulos
    if df.isnull().sum().sum() > 0:
        print("\n[ADVERTENCIA] Se encontraron valores nulos en el dataset. Se eliminarán las filas afectadas.")
        df.dropna(inplace=True)
    
    if df.empty:
        print("\n[ERROR CRÍTICO] El dataset está vacío después de la limpieza. No se puede continuar.")
        sys.exit()

    # --- Preprocesamiento (sin cambios, pero con más feedback) ---
    print("\n[Fase 2 de 4] Preprocesando los datos...")
    X = df.drop(['filename', 'label'], axis=1)
    y = df['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Preprocesamiento completado (Codificación y Escalado).")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print("División de datos en entrenamiento (80%) y prueba (20%) completada.")

    # --- Entrenamiento del Modelo con Indicador de Progreso ---
    print("\n[Fase 3 de 4] Entrenando el modelo RandomForest...")
    print("El entrenamiento puede tardar unos segundos. Se mostrará el progreso a continuación:")
    
    # --- MEJORA 2: Se añade 'verbose=1' para mostrar el progreso ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)
    print("\n¡Nuevo modelo entrenado exitosamente!")

    # --- Evaluación y Guardado ---
    print("\n[Fase 4 de 4] Evaluando y guardando el nuevo modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n----------------- RESULTADOS DEL NUEVO MODELO -----------------")
    print(f"Precisión (Accuracy) final: {accuracy * 100:.2f}%")
    print("---------------------------------------------------------------")

    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    joblib.dump(model, os.path.join(MODELS_DIR, 'modelo_mejorado.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_mejorado.joblib'))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'encoder_mejorado.joblib'))
    
    print(f"\n¡PROCESO COMPLETADO! Nuevo modelo y preprocesadores guardados en la carpeta '{MODELS_DIR}'.")
    print("El siguiente paso es actualizar 'app.py' para usar estos nuevos archivos.")

if __name__ == '__main__':
    main()