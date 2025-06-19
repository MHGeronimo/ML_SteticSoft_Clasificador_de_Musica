# src/pages/2_Análisis_del_Modelo.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# --- MANEJO DE RUTAS ---
ROOT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# --- FUNCIONES ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / 'modelo_genero_musical.joblib')
    label_encoder = joblib.load(MODELS_DIR / 'encoder.joblib')
    return model, label_encoder

@st.cache_data
def get_test_data():
    df = pd.read_csv(DATA_DIR / 'features_3_sec.csv')
    X = df.drop(['filename', 'length', 'label'], axis=1)
    y = df['label']
    
    y_encoded = LabelEncoder().fit_transform(y)
    X_scaled = StandardScaler().fit_transform(X)

    _, X_test, _, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    return X_test, y_test

# --- CARGA ---
try:
    model, label_encoder = load_artifacts()
    X_test, y_test = get_test_data()
    LOAD_SUCCESS = True
except FileNotFoundError:
    st.error("Error al cargar archivos. Asegúrate que 'models' y 'data' estén en la raíz.")
    LOAD_SUCCESS = False

# --- INTERFAZ ---
st.title("📊 Análisis de Rendimiento del Modelo")
st.write("Aquí evaluamos el rendimiento de nuestro `RandomForestClassifier` con los datos de prueba.")

if LOAD_SUCCESS:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.header(f"Precisión (Accuracy) del Modelo")
    st.metric(label="Accuracy", value=f"{accuracy:.2%}")
    st.progress(accuracy)

    st.header("Reporte de Clasificación")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.header("Matriz de Confusión")
    st.write("Esta matriz muestra en qué géneros se confunde el modelo. Los números en la diagonal principal son las predicciones correctas.")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Género Real')
    st.pyplot(fig)