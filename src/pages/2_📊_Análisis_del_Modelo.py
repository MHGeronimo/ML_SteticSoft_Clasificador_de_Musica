# src/pages/2_Análisis_del_Modelo.py (Versión actualizada para el modelo mejorado)

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# --- MANEJO DE RUTAS ROBUSTO ---
ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models_mejorados"
DATA_DIR = ROOT_DIR / "data"

# --- FUNCIONES ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / 'modelo_mejorado.joblib')
    label_encoder = joblib.load(MODELS_DIR / 'encoder_mejorado.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler_mejorado.joblib')
    return model, scaler, label_encoder

@st.cache_data
def get_test_data(_scaler, _label_encoder):
    """
    Carga el dataset, lo procesa y devuelve los datos de prueba.
    """
    # --- CORRECCIÓN: Apuntar al nombre correcto del archivo CSV ---
    df = pd.read_csv(DATA_DIR / 'features_refactored_3_sec.csv')
    
    X = df.drop(['filename', 'label'], axis=1)
    y = df['label']
    
    y_encoded = _label_encoder.transform(y)
    X_scaled = _scaler.transform(X)

    _, X_test, _, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    return X_test, y_test

# --- CARGA ---
try:
    model, scaler, label_encoder = load_artifacts()
    X_test, y_test = get_test_data(scaler, label_encoder)
    LOAD_SUCCESS = True
except FileNotFoundError:
    # --- CORRECCIÓN: Mensaje de error actualizado ---
    st.error("Error al cargar archivos. Asegúrate de que la carpeta 'models_mejorados' y el archivo 'features_refactored_3_sec.csv' existan.")
    LOAD_SUCCESS = False

# --- INTERFAZ ---
st.set_page_config(page_title="Análisis del Modelo", page_icon="📊")
st.title("📊 Análisis de Rendimiento del Modelo")
st.write("Aquí evaluamos el rendimiento de nuestro `RandomForestClassifier` (entrenado con 59 características).")

if LOAD_SUCCESS:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.header(f"Precisión (Accuracy) del Modelo")
    st.metric(label="Accuracy", value=f"{accuracy:.2%}")
    st.progress(accuracy)

    st.header("Reporte de Clasificación")
    st.write("Muestra las métricas de `precision`, `recall` y `f1-score` para cada género.")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.header("Matriz de Confusión")
    st.write("Esta matriz visualiza en qué géneros se confunde más el modelo. La diagonal principal representa las predicciones correctas.")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Género Real')
    st.pyplot(fig)
