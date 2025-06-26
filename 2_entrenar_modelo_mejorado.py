import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

def train_model():
    """
    Función principal para entrenar el modelo.
    """
    CSV_PATH = os.path.join('data', 'features_refactored_3_sec.csv')
    MODELS_DIR = 'models_mejorados'

    print("--- INICIANDO PROCESO DE ENTRENAMIENTO DEL MODELO ---")

    # --- Fase 1: Carga y Validación del Dataset ---
    print("\n[Paso 1 de 4] Cargando y validando el dataset...")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"-> Dataset '{CSV_PATH}' cargado exitosamente.")
    except FileNotFoundError:
        print(f"\n[ERROR CRÍTICO] No se encontró el archivo del dataset: '{CSV_PATH}'")
        print("-> Asegúrate de ejecutar primero '1_crear_dataset_mejorado.py'.")
        sys.exit()

    # Validar y limpiar datos nulos
    if df.isnull().values.any():
        print("-> [ADVERTENCIA] Se encontraron valores nulos. Eliminando filas afectadas...")
        df.dropna(inplace=True)

    if df.empty:
        print("\n[ERROR CRÍTICO] El dataset está vacío. No se puede continuar.")
        sys.exit()

    # --- Fase 2: Preprocesamiento de Datos ---
    print("\n[Paso 2 de 4] Preprocesando los datos...")
    X = df.drop(['filename', 'label'], axis=1)
    y = df['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("-> Codificación de etiquetas y escalado de características completado.")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print("-> División de datos en entrenamiento (80%) y prueba (20%) completada.")

    # --- Fase 3: Entrenamiento del Modelo ---
    print("\n[Paso 3 de 4] Entrenando el modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("-> ¡Modelo entrenado exitosamente!")

    # --- Fase 4: Evaluación y Guardado ---
    print("\n[Paso 4 de 4] Evaluando y guardando los artefactos...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n----------------- RESULTADOS DEL MODELO -----------------")
    print(f" Precisión (Accuracy) Final: {accuracy * 100:.2f}%")
    print("---------------------------------------------------------")
    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'modelo_mejorado.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_mejorado.joblib'))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'encoder_mejorado.joblib'))

    print(f"\n¡PROCESO COMPLETADO! Modelo y preprocesadores guardados en la carpeta '{MODELS_DIR}'.")
    print("Ahora puedes ejecutar la aplicación de Streamlit: streamlit run src/app.py")

# --- CORRECCIÓN CLAVE ---
# Se llama a la función train_model() para que el script se ejecute.
if __name__ == '__main__':
    train_model()
