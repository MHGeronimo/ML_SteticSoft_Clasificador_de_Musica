# src/pages/3_Sobre_el_Proyecto.py

import streamlit as st

st.set_page_config(page_title="Sobre el Proyecto", page_icon="📄")

st.title("📄 Sobre el Proyecto")

st.markdown("""
### Objetivo
El propósito de este proyecto es aplicar técnicas de Machine Learning para construir un sistema capaz de clasificar música en diferentes géneros a partir de sus características de audio. Este clasificador puede ser útil para la organización automática de librerías musicales, sistemas de recomendación y análisis musicológico.

### Dataset
Utilizamos el dataset **GTZAN Genre Collection**, una colección popular para tareas de clasificación de géneros musicales. Contiene 1000 pistas de audio de 30 segundos cada una, divididas equitativamente en 10 géneros.
- **Fuente:** [GTZAN Genre Collection en Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Géneros (10):** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.

### Tecnologías Utilizadas
- **Lenguaje:** Python
- **Librerías de Machine Learning:** Scikit-learn
- **Procesamiento de Audio:** Librosa
- **Manipulación de Datos:** Pandas, NumPy
- **Framework de la Aplicación Web:** Streamlit
- **Visualización:** Matplotlib, Seaborn

---

### Equipo de Trabajo
- Eylin Alejandra Mora Arboleda
- Yulitza Tatiana Caicedo Mosquera
- Juan Esteban Gómez Londoño
- Geronimo Martinez Higuita
""")