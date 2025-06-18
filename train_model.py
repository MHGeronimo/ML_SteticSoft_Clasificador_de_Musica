# train_model.py

# 1. Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os 

# 2. Cargar los datos
print("Cargando datos...")
df = pd.read_csv('data/features_3_sec.csv')

# 3. Exploración inicial (opcional pero recomendado)
print("Forma del dataset:", df.shape)
print("Primeras filas de datos:")
print(df.head())
print("\nDistribución de géneros musicales:")
print(df['label'].value_counts())

# 4. Preparación de los datos
# Eliminar columnas no necesarias para el modelo
X = df.drop(['filename', 'length', 'label'], axis=1)
y = df['label']

# Codificar las etiquetas de texto (géneros) a números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Escalar las características numéricas
# Es crucial para que el modelo no se incline por características con valores más grandes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nDatos listos para el entrenamiento.")

# 5. Dividir los datos en conjuntos de entrenamiento y prueba
# 80% para entrenar, 20% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y_encoded, 
    test_size=0.2, 
    random_state=42, # Para resultados reproducibles
    stratify=y_encoded # Mantiene la proporción de géneros en ambos conjuntos
)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} filas")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} filas")

# 6. Crear y entrenar el modelo
print("\nEntrenando el modelo RandomForest...")
# Usamos RandomForest porque es potente y versátil
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("¡Modelo entrenado exitosamente!")

# 7. Evaluar el rendimiento del modelo
print("\nEvaluando el modelo...")
y_pred = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy * 100:.2f}%")

# Mostrar un reporte detallado
print("\nReporte de Clasificación:")
original_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
print(classification_report(y_test, y_pred, target_names=original_labels))


# 8. Guardar el modelo y los objetos de preprocesamiento en la carpeta /models
print("\nGuardando el modelo y los preprocesadores...")

# Definir el nombre de la carpeta
MODELS_DIR = 'models'

# Crear la carpeta si no existe
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Carpeta '{MODELS_DIR}' creada.")

# Guardar archivos dentro de la carpeta 'models'
# Usamos os.path.join para construir la ruta de forma segura
joblib.dump(model, os.path.join(MODELS_DIR, 'modelo_genero_musical.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'encoder.joblib'))

print(f"\n¡Proceso completado! Modelo y preprocesadores guardados en la carpeta '{MODELS_DIR}'.")