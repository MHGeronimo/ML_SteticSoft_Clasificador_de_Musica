# src/pages/3__Explorar_Canciones.py (O el nombre que le hayas puesto)

import streamlit as st
from urllib.parse import quote
from utils import GENRE_DATA # Importamos la nueva estructura de datos

st.set_page_config(page_title="Explorar Géneros", page_icon="🎧", layout="wide")

st.title("🎧 Explorar Géneros Musicales")
st.write("Selecciona un género para descubrir canciones clave y los artistas más representativos.")

# --- INTERFAZ DE LA PÁGINA ---
genre_list = list(GENRE_DATA.keys())
selected_genre = st.selectbox("Elige un género:", genre_list)

if selected_genre:
    
    # Usamos pestañas para organizar el contenido
    song_tab, artist_tab = st.tabs(["🎵 Canciones Clave", "🎤 Artistas Representativos"])
    
    with song_tab:
        st.subheader(f"10 Canciones Esenciales de {selected_genre.capitalize()}")
        
        songs = GENRE_DATA[selected_genre]["songs"]
        
        for song in songs:
            # Crear un enlace de búsqueda en YouTube
            query = quote(song)
            youtube_url = f"https://www.youtube.com/results?search_query={query}"
            
            # Mostrar la canción como un enlace clickable
            st.markdown(f"- [{song}]({youtube_url})")

    with artist_tab:
        st.subheader(f"30 Artistas Representativos de {selected_genre.capitalize()}")
        st.write("Ordenados de mayor a menor reconocimiento e influencia en el género.")
        
        artists = GENRE_DATA[selected_genre]["artists"]
        
        # Dividir la lista de artistas en 3 columnas para una mejor visualización
        col1, col2, col3 = st.columns(3)
        
        for index, artist in enumerate(artists):
            if index % 3 == 0:
                col1.write(f"{index + 1}. {artist}")
            elif index % 3 == 1:
                col2.write(f"{index + 1}. {artist}")
            else:
                col3.write(f"{index + 1}. {artist}")